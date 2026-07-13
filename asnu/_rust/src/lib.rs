use std::collections::{HashMap, HashSet};

use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::seq::SliceRandom;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2} ;


/// Per-phase diagnostics.
#[derive(Default)]
struct PhaseCounters {
    primary: u64,
    reciprocity: u64,
    trans_int: u64,
    trans_ext: u64,
    pairs_touched: u64,
}

impl PhaseCounters {
    fn total(&self) -> u64 {
        self.primary + self.reciprocity + self.trans_int + self.trans_ext
    }
}

/// Owns all mutable graph state plus the parameters needed for the
/// reciprocity / transitivity follow-ons, so the "insert edge, maybe
/// reciprocate, maybe close triangles" sequence exists in exactly one place.
///
/// `link_counts` is the single source of truth for per-group-pair budgets —
/// there is no separate `num_links` local to keep in sync, which removes a
/// whole class of counter-drift bugs (the old `dst_id == src_id` special
/// cases fall out automatically).
struct EdgeBuilder<'a> {
    edges: HashSet<(i64, i64)>,
    adjacency: HashMap<i64, Vec<i64>>,
    /// Incoming edges, so transitivity can scan the UNDIRECTED neighbourhood
    /// out[d] ∪ in[d] (matches NetworkX's neighbors(d)). Only maintained when
    /// transitivity is active, so the baseline pays nothing.
    in_adjacency: HashMap<i64, Vec<i64>>,
    new_edges: Vec<(i64, i64)>,
    /// (src_group, dst_group) -> number of links created so far.
    link_counts: HashMap<(i64, i64), i64>,

    max_links: &'a HashMap<(i64, i64), i64>,
    nodes_to_group: &'a HashMap<i64, i64>,
    /// node -> community id (group ignored — option 2).
    node_to_community: HashMap<i64, i64>,

    reciprocity_p: f64,
    int_trans_p: f64,
    ext_trans_p: f64,
    use_transitivity: bool,
}

impl EdgeBuilder<'_> {
    fn links(&self, pair: (i64, i64)) -> i64 {
        *self.link_counts.get(&pair).unwrap_or(&0)
    }

    fn max(&self, pair: (i64, i64)) -> i64 {
        *self.max_links.get(&pair).unwrap_or(&0)
    }

    /// Insert edge s→d attributed to `pair`, updating adjacency, output list
    /// and the pair's budget counter. Returns false if it already exists.
    fn insert(&mut self, s: i64, d: i64, pair: (i64, i64)) -> bool {
        if s == d || !self.edges.insert((s, d)) {
            return false;
        }
        self.adjacency.entry(s).or_default().push(d);
        if self.use_transitivity {
            self.in_adjacency.entry(d).or_default().push(s);
        }
        self.new_edges.push((s, d));
        *self.link_counts.entry(pair).or_insert(0) += 1;
        true
    }

    /// Register a pre-existing edge (multiplex pre-seeding): counts toward
    /// the budget and the adjacency, but is NOT added to `new_edges`.
    fn seed(&mut self, s: i64, d: i64) {
        self.edges.insert((s, d));
        self.adjacency.entry(s).or_default().push(d);
        if self.use_transitivity {
            self.in_adjacency.entry(d).or_default().push(s);
        }
        if let (Some(&sg), Some(&dg)) = (self.nodes_to_group.get(&s), self.nodes_to_group.get(&d)) {
            *self.link_counts.entry((sg, dg)).or_insert(0) += 1;
        }
    }

    /// Reciprocity roll for a freshly created s(group sg) → d(group dg) edge.
    fn maybe_reciprocate(&mut self, rng: &mut ThreadRng, s: i64, d: i64, sg: i64, dg: i64, counter: &mut u64) {
        if rng.gen::<f64>() >= self.reciprocity_p {
            return;
        }
        let rev = (dg, sg);
        if self.links(rev) < self.max(rev) && self.insert(d, s, rev) {
            *counter += 1;
        }
    }

    /// Triadic closures through pivot d: for each undirected neighbour n of d,
    /// roll int_trans_p when n shares d's community id, ext_trans_p otherwise
    /// (two unknowns are NOT treated as equal). Stops once `budget_pair`
    /// reaches `target`. Each closure gets its own reciprocity roll.
    fn apply_transitivity(
        &mut self,
        rng: &mut ThreadRng,
        s: i64,
        d: i64,
        src_group: i64,
        budget_pair: (i64, i64),
        target: i64,
        c: &mut PhaseCounters,
    ) {
        if !self.use_transitivity {
            return;
        }
        // Undirected neighbourhood of d, deduplicated. (Snapshot: closures
        // created below must not extend the scan, matching the original.)
        let mut neighbors: Vec<i64> = self.adjacency.get(&d).cloned().unwrap_or_default();
        if let Some(ins) = self.in_adjacency.get(&d) {
            neighbors.extend(ins.iter().copied());
        }
        let mut seen = HashSet::new();
        neighbors.retain(|&x| seen.insert(x));

        let d_comm = self.node_to_community.get(&d).copied();

        for n in neighbors {
            if self.links(budget_pair) >= target {
                break;
            }
            if s == n {
                continue;
            }
            let internal = matches!(
                (self.node_to_community.get(&n), d_comm),
                (Some(&nc), Some(dc)) if nc == dc
            );
            let p = if internal { self.int_trans_p } else { self.ext_trans_p };
            if rng.gen::<f64>() >= p {
                continue;
            }
            let Some(&n_group) = self.nodes_to_group.get(&n) else { continue };
            let pair = (src_group, n_group);
            let Some(&max_l) = self.max_links.get(&pair) else { continue };
            if self.links(pair) < max_l && self.insert(s, n, pair) {
                if internal { c.trans_int += 1 } else { c.trans_ext += 1 }
                self.maybe_reciprocate(rng, s, n, src_group, n_group, &mut c.reciprocity);
            }
        }
    }

    /// Full edge-creation step: primary edge s→d, then reciprocity and
    /// transitivity follow-ons. Returns true if the primary edge was created.
    fn create_edge(
        &mut self,
        rng: &mut ThreadRng,
        s: i64,
        d: i64,
        src_group: i64,
        dst_group: i64,
        target: i64,
        c: &mut PhaseCounters,
    ) -> bool {
        let pair = (src_group, dst_group);
        if !self.insert(s, d, pair) {
            return false;
        }
        c.primary += 1;
        self.maybe_reciprocate(rng, s, d, src_group, dst_group, &mut c.reciprocity);
        self.apply_transitivity(rng, s, d, src_group, pair, target, c);
        true
    }
}

#[pyfunction]
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None, node_coordinates=None, internal_transitivity_p=-1.0, external_transitivity_p=-1.0))]
#[allow(clippy::too_many_arguments)]
fn run_edge_creation(
    group_pairs: Vec<(i64, i64, i64)>,
    valid_communities_map: HashMap<(i64, i64), Vec<i64>>,
    maximum_num_links: HashMap<(i64, i64), i64>,
    communities_to_nodes: HashMap<(i64, i64), Vec<i64>>,
    nodes_to_group: HashMap<i64, i64>,
    fraction: f64,
    reciprocity_p: f64,
    transitivity_p: f64,
    pa_scope: String,
    number_of_communities: i64,
    bridge_probability: f64,
    pre_existing_edges: Option<Vec<(i64, i64)>>,
    node_coordinates: Option<HashMap<i64, f64>>,
    internal_transitivity_p: f64,
    external_transitivity_p: f64,
) -> PyResult<(Vec<(i64, i64)>, Vec<(i64, i64, i64)>)> {
    let mut rng = thread_rng();

    // Effective per-side transitivity (negative falls back to the scalar,
    // so existing call sites keep working).
    let int_trans_p = if internal_transitivity_p < 0.0 { transitivity_p } else { internal_transitivity_p };
    let ext_trans_p = if external_transitivity_p < 0.0 { transitivity_p } else { external_transitivity_p };
    let use_transitivity = int_trans_p > 0.0 || ext_trans_p > 0.0;

    let node_to_community: HashMap<i64, i64> = if use_transitivity {
        communities_to_nodes
            .iter()
            .flat_map(|(&(comm_id, _), nodes)| nodes.iter().map(move |&n| (n, comm_id)))
            .collect()
    } else {
        HashMap::new()
    };

    let mut b = EdgeBuilder {
        edges: HashSet::new(),
        adjacency: HashMap::new(),
        in_adjacency: HashMap::new(),
        new_edges: Vec::new(),
        // Pre-insert 0 for every known pair so the returned triples cover
        // all budgeted pairs, including untouched ones.
        link_counts: maximum_num_links.keys().map(|&k| (k, 0)).collect(),
        max_links: &maximum_num_links,
        nodes_to_group: &nodes_to_group,
        node_to_community,
        reciprocity_p,
        int_trans_p,
        ext_trans_p,
        use_transitivity,
    };

    // Multiplex pre-seeding: counts toward budgets, not toward new_edges.
    let pre_edge_count = pre_existing_edges.as_ref().map_or(0, |v| v.len());
    for &(s, d) in pre_existing_edges.iter().flatten() {
        b.seed(s, d);
    }
    if pre_edge_count > 0 {
        println!("  Rust: initialized with {pre_edge_count} pre-existing edges");
    }

    // Phase B precomputation: src nodes sorted by coordinate, dst communities
    // sorted by centroid. Ring search finds nearest communities then picks a
    // random node from each — spreading degree load across all nodes rather
    // than targeting edge-nearest ones.
    let mut group_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    let mut group_comm_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    if let Some(ref nc) = node_coordinates {
        let coord = |n: i64| *nc.get(&n).unwrap_or(&0.5);
        for (&(comm_id, gid), nodes) in &communities_to_nodes {
            group_sorted
                .entry(gid)
                .or_default()
                .extend(nodes.iter().map(|&n| (coord(n), n)));
            if !nodes.is_empty() {
                let centroid = nodes.iter().map(|&n| coord(n)).sum::<f64>() / nodes.len() as f64;
                group_comm_sorted.entry(gid).or_default().push((centroid, comm_id));
            }
        }
        for v in group_sorted.values_mut().chain(group_comm_sorted.values_mut()) {
            v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        println!("  Phase B: built sorted arrays for {} groups", group_sorted.len());
    }

    let total_pairs = group_pairs.len();
    let mut phase_a = PhaseCounters::default();
    let mut phase_b = PhaseCounters::default();

    // Caches shared across pairs.
    let mut src_node_cache: HashMap<(i64, i64), Vec<i64>> = HashMap::new();
    let mut popularity_pool: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    // ── Phase A: community-based edge creation ──────────────────────────────
    // Communities are iterated sequentially (shuffled once per pair) so each
    // community exhausts a proportional quota before moving to the next.
    // This concentrates edges within communities, raising transitivity.
    for (pair_idx, &(src_id, dst_id, target)) in group_pairs.iter().enumerate() {
        if (pair_idx + 1) % 5000 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let pair = (src_id, dst_id);
        if b.links(pair) >= target {
            continue;
        }
        let Some(communities) = valid_communities_map.get(&pair).filter(|v| !v.is_empty()) else {
            continue;
        };

        let edges_before = b.new_edges.len();

        // Deduplicate (the map may carry duplicates for weighting) and shuffle.
        let mut comm_order: Vec<i64> = {
            let mut seen = HashSet::new();
            communities.iter().filter(|&&c| seen.insert(c)).copied().collect()
        };
        comm_order.shuffle(&mut rng);
        let n_comms = comm_order.len();

        const MAX_PASSES: i64 = 3;
        let mut pass = 0;

        'outer: while b.links(pair) < target && pass < MAX_PASSES {
            pass += 1;
            for &community_id in &comm_order {
                let remaining = target - b.links(pair);
                if remaining <= 0 {
                    break 'outer;
                }
                let quota = (remaining as usize).div_ceil(n_comms).max(1);

                let src_key = (community_id, src_id);
                let src_nodes = src_node_cache
                    .entry(src_key)
                    .or_insert_with(|| communities_to_nodes.get(&src_key).cloned().unwrap_or_default());
                if src_nodes.is_empty() {
                    continue;
                }

                // Bridge or normal dst community.
                let dst_community = if bridge_probability > 0.0
                    && number_of_communities > 1
                    && rng.gen::<f64>() < bridge_probability
                {
                    let direction: i64 = if rng.gen::<bool>() { 1 } else { -1 };
                    (community_id + direction).rem_euclid(number_of_communities)
                } else {
                    community_id
                };

                // Popularity pool for (dst_community, dst_group): a random
                // `fraction`-sized sample; PA later grows it with repeats.
                let pool_key = (dst_community, dst_id);
                if !popularity_pool.contains_key(&pool_key) {
                    let mut pool = communities_to_nodes.get(&pool_key).cloned().unwrap_or_default();
                    if !pool.is_empty() {
                        let sample_size = ((pool.len() as f64) * fraction).ceil() as usize;
                        pool.shuffle(&mut rng);
                        pool.truncate(sample_size.min(pool.len()));
                    }
                    popularity_pool.insert(pool_key, pool);
                }
                if popularity_pool[&pool_key].is_empty() {
                    continue;
                }

                // Create up to `quota` edges within this community.
                let mut created = 0usize;
                let max_local = quota * 3;

                for _ in 0..max_local {
                    if created >= quota || b.links(pair) >= target {
                        break;
                    }
                    let s = src_nodes[rng.gen_range(0..src_nodes.len())];
                    let d = {
                        let pool = &popularity_pool[&pool_key];
                        pool[rng.gen_range(0..pool.len())]
                    };

                    if !b.create_edge(&mut rng, s, d, src_id, dst_id, target, &mut phase_a) {
                        continue;
                    }
                    created += 1;

                    // Preferential attachment: occasionally re-add d to pools
                    // so it gets picked more often.
                    if fraction != 1.0 && rng.gen::<f64>() > fraction {
                        if pa_scope == "global" {
                            for comm_id in 0..number_of_communities {
                                if rng.gen::<f64>() < fraction / number_of_communities as f64 {
                                    if let Some(p) = popularity_pool.get_mut(&(comm_id, dst_id)) {
                                        p.push(d);
                                    }
                                }
                            }
                        } else if rng.gen::<f64>() > fraction {
                            if let Some(p) = popularity_pool.get_mut(&pool_key) {
                                p.push(d);
                                if let Some(nodes) = communities_to_nodes.get(&pool_key) {
                                    if !nodes.is_empty() {
                                        p.push(nodes[rng.gen_range(0..nodes.len())]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if b.new_edges.len() > edges_before {
            phase_a.pairs_touched += 1;
        }
    }

    // ── Phase B: spatial ring search for remaining budget ────────────────────
    // For each pair still under budget, finds nearest dst communities by
    // centroid and picks a random node — fills cross-block pairs left by A.
    const PHASE_B_COMM_WINDOW: usize = 200;

    for (pair_idx, &(src_id, dst_id, target)) in group_pairs.iter().enumerate() {
        if (pair_idx + 1) % 5000 == 0 {
            println!("Phase B: pair {} of {}", pair_idx + 1, total_pairs);
        }
        let pair = (src_id, dst_id);
        if b.links(pair) >= target {
            continue;
        }
        let (Some(src_sorted), Some(dst_comm_sorted)) =
            (group_sorted.get(&src_id), group_comm_sorted.get(&dst_id))
        else {
            continue;
        };

        let n_dst_comm = dst_comm_sorted.len();
        let win = PHASE_B_COMM_WINDOW.min(n_dst_comm);
        let n_src = src_sorted.len();
        if n_src == 0 || win == 0 {
            continue;
        }

        let edges_before = b.new_edges.len();
        let mut src_indices: Vec<usize> = (0..n_src).collect();

        loop {
            let remaining = target - b.links(pair);
            if remaining <= 0 {
                break;
            }
            src_indices.shuffle(&mut rng);
            let edges_per_src = (remaining as usize).div_ceil(n_src).max(1).min(win);
            let mut made_progress = false;

            for &si in &src_indices {
                if b.links(pair) >= target {
                    break;
                }
                let (theta_s, s) = src_sorted[si];
                let center = dst_comm_sorted.partition_point(|&(c, _)| c < theta_s);
                let mut found = 0usize;

                'ring: for delta in 0..win {
                    if found >= edges_per_src {
                        break 'ring;
                    }
                    let j1 = (center + delta) % n_dst_comm;
                    let j2 = (center + n_dst_comm - delta - 1) % n_dst_comm;
                    for &j in &[j1, j2] {
                        if found >= edges_per_src || b.links(pair) >= target {
                            break;
                        }
                        let (_, comm_id) = dst_comm_sorted[j];
                        let Some(dst_nodes) = communities_to_nodes.get(&(comm_id, dst_id)) else { continue };
                        if dst_nodes.is_empty() {
                            continue;
                        }
                        let d = dst_nodes[rng.gen_range(0..dst_nodes.len())];
                        if b.create_edge(&mut rng, s, d, src_id, dst_id, target, &mut phase_b) {
                            found += 1;
                            made_progress = true;
                        }
                    }
                }
            }
            if !made_progress {
                break;
            }
        }

        if b.new_edges.len() > edges_before {
            phase_b.pairs_touched += 1;
        }
    }

    // ── Phase diagnostics report ─────────────────────────────────────────────
    let grand_total = phase_a.total() + phase_b.total();
    let pct = |x: u64| if grand_total == 0 { 0.0 } else { 100.0 * x as f64 / grand_total as f64 };
    let report = |name: &str, c: &PhaseCounters| {
        println!("│ Phase {name}");
        println!("│   primary          : {:>10}   ({:>5.1}%)", c.primary, pct(c.primary));
        println!("│   reciprocity      : {:>10}   ({:>5.1}%)", c.reciprocity, pct(c.reciprocity));
        println!("│   transitivity int : {:>10}   ({:>5.1}%)", c.trans_int, pct(c.trans_int));
        println!("│   transitivity ext : {:>10}   ({:>5.1}%)", c.trans_ext, pct(c.trans_ext));
        println!("│   subtotal         : {:>10}   ({:>5.1}%)", c.total(), pct(c.total()));
        println!("│   pairs touched    : {:>10} / {total_pairs}", c.pairs_touched);
    };

    println!("\n┌─ Edge creation diagnostics ─────────────────────────────");
    report("A (community-based)", &phase_a);
    report("B (ring search)", &phase_b);
    println!("│ Transitivity gate : community-id  (int_p={int_trans_p:.2}, ext_p={ext_trans_p:.2})");
    println!("│ Total new edges   : {grand_total:>10}");
    if grand_total as usize != b.new_edges.len() {
        println!(
            "│ ⚠ counter mismatch: counted={} actual={} (diff={})",
            grand_total,
            b.new_edges.len(),
            b.new_edges.len() as i64 - grand_total as i64
        );
    }
    println!("└─────────────────────────────────────────────────────────");

    let links_out: Vec<(i64, i64, i64)> = b.link_counts.into_iter().map(|((s, d), c)| (s, d, c)).collect();
    Ok((b.new_edges, links_out))
}

#[pyfunction]
#[pyo3(signature = (
    assignments, node_groups, budget,
    n_groups, n_communities,
    n_iterations = 100_000,
    loss_goal = 0.0,
    overshoot_penalty = 1.0,
    seed = 42,
))]
fn refine_communities_move<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    mut n_communities: usize, // Made mutable to allow growth
    n_iterations: usize,    
    loss_goal: f64,
    overshoot_penalty: f64,
    seed: u64,
)  -> PyResult<(Bound<'py, PyArray1<i64>>, f64)> {
    let assigns = assignments.as_array();
    let groups = node_groups.as_array();
    let n = assigns.len();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut current: Vec<usize> = (0..n).map(|i| assigns[i] as usize).collect();

    // comp[c] = {group: count}. Using a Vec of HashMaps.
    let mut comp: Vec<HashMap<usize, i64>> = (0..n_communities).map(|_| HashMap::new()).collect();
    for i in 0..n {
        let c = current[i];
        let g = groups[i] as usize;
        if c < n_communities && g < n_groups {
            *comp[c].entry(g).or_insert(0) += 1;
        }
    }

    let mut achieved: HashMap<(usize, usize), i64> = HashMap::new();
    for c_comp in &comp {
        for (&g, &cg) in c_comp {
            for (&h, &ch) in c_comp {
                *achieved.entry((g, h)).or_insert(0) += cg * ch;
            }
        }
    }

    let pair_cost = |achieved_val: i64, budget_val: i64| -> f64 {
        let d = achieved_val as f64 - budget_val as f64;
        if d > 0.0 { overshoot_penalty * d } else { -d }
    };

    let mut current_loss = 0.0;
    // Initial loss calculation
    for (&(g, h), &av) in &achieved {
        let bv = budget.get(&(g as i64, h as i64)).copied().unwrap_or(0);
        current_loss += pair_cost(av, bv);
    }
    for (&(g, h), &bv) in &budget {
        if !achieved.contains_key(&(g as usize, h as usize)) {
            current_loss += pair_cost(0, bv);
        }
    }

    let mut accepted = 0usize;
    let report_every = (n_iterations / 10).max(1);

    for iter in 0..n_iterations {
        let i = rng.gen_range(0..n);
        let g = groups[i] as usize;
        let c_old = current[i];
        
        // // Pick a target community: existing OR one brand new potential ID
        // let c_new = rng.gen_range(0..=n_communities);

        // Pick a target community: existing communities only (no growth)
        let c_new = rng.gen_range(0..n_communities);

        if c_old == c_new { continue; }

        // If we chose a new community, ensure the comp vector can hold it
        if c_new == n_communities {
            // We don't actually push to comp yet, just prepare the logic
        }

        let mut delta_loss = 0.0;
        let mut affected_updates: Vec<((usize, usize), i64)> = Vec::new();

        // The only group whose count changes is 'g'.
        // This affects pairs (g, h) and (h, g) for all h present in c_old or c_new.
        let mut affected_groups: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &h in comp[c_old].keys() { affected_groups.insert(h); }
        if c_new < n_communities {
            for &h in comp[c_new].keys() { affected_groups.insert(h); }
        }
        affected_groups.insert(g); 

        for &h in &affected_groups {
            let c_old_g_old = *comp[c_old].get(&g).unwrap_or(&0);
            let c_old_h_old = *comp[c_old].get(&h).unwrap_or(&0);
            
            let c_new_g_old = if c_new < n_communities { *comp[c_new].get(&g).unwrap_or(&0) } else { 0 };
            let c_new_h_old = if c_new < n_communities { *comp[c_new].get(&h).unwrap_or(&0) } else { 0 };

            // After move: c_old[g] decreases, c_new[g] increases
            let c_old_g_new = c_old_g_old - 1;
            let c_new_g_new = c_new_g_old + 1;
            
            // h doesn't change, but if h == g, we use the new values
            let c_old_h_new = if h == g { c_old_g_new } else { c_old_h_old };
            let c_new_h_new = if h == g { c_new_g_new } else { c_new_h_old };

            let old_contrib = (c_old_g_old * c_old_h_old) + (c_new_g_old * c_new_h_old);
            let new_contrib = (c_old_g_new * c_old_h_new) + (c_new_g_new * c_new_h_new);
            let delta = new_contrib - old_contrib;

            if delta != 0 {
                let keys = if g == h { vec![(g, g)] } else { vec![(g, h), (h, g)] };
                for (ga, ha) in keys {
                    let av_old = *achieved.get(&(ga, ha)).unwrap_or(&0);
                    let bv = budget.get(&(ga as i64, ha as i64)).copied().unwrap_or(0);
                    let av_new = av_old + delta;
                    delta_loss += pair_cost(av_new, bv) - pair_cost(av_old, bv);
                    affected_updates.push(((ga, ha), delta));
                }
            }
        }

        if delta_loss < 0.0 {
            // Apply Move
            if c_new == n_communities {
                comp.push(HashMap::new());
                n_communities += 1;
            }

            // Update comp
            let count_old = comp[c_old].get_mut(&g).unwrap();
            *count_old -= 1;
            if *count_old == 0 { comp[c_old].remove(&g); }
            *comp[c_new].entry(g).or_insert(0) += 1;

            // Update achieved
            for ((ga, ha), d) in affected_updates {
                let entry = achieved.entry((ga, ha)).or_insert(0);
                *entry += d;
                if *entry == 0 { achieved.remove(&(ga, ha)); }
            }

            current[i] = c_new;
            current_loss += delta_loss;
            accepted += 1;
            if current_loss <= loss_goal {
            println!("Reached loss goal at iter {}: loss={:.2}", iter + 1, current_loss);
            break;
            }
        }

        if (iter + 1) % report_every == 0 {
            println!("Iter {}: loss={:.2}, communities={}", iter+1, current_loss, n_communities);
        }
    }

    let result: Vec<i64> = current.iter().map(|&c| c as i64).collect();
    let arr = PyArray1::from_owned_array_bound(py, Array1::from(result));
    Ok((arr, current_loss))
}


/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    m.add_function(wrap_pyfunction!(refine_communities_move, m)?)?;
    Ok(())
}