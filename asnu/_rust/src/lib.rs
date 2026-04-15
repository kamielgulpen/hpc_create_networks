use std::collections::{HashMap, HashSet};

use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1, PyReadwriteArray2};
use pyo3::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::seq::SliceRandom;

/// Port of the "process nodes" loop from populate_communities() (probability mode).
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, community_composition, community_sizes, group_exposure, ideal, target_counts=None, total_nodes=0))]
fn process_nodes<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    mut community_composition: PyReadwriteArray2<'py, f64>,
    mut community_sizes: PyReadwriteArray1<'py, i32>,
    mut group_exposure: PyReadwriteArray2<'py, f64>,
    ideal: PyReadonlyArray2<'py, f64>,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let all_nodes = all_nodes.as_array();
    let node_groups = node_groups.as_array();
    let ideal = ideal.as_array();

    let num_communities = community_composition.as_array().shape()[0];
    let n_groups = community_composition.as_array().shape()[1];

    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());

    let mut rng = thread_rng();
    let mut assignments: Vec<i64> = Vec::with_capacity(total_nodes);

    let mut distances = vec![0.0f64; num_communities];
    let mut hyp_row = vec![0.0f64; n_groups];

    for node_idx in 0..total_nodes {
        let group = node_groups[node_idx] as usize;

        let comp = community_composition.as_array();
        let ge = group_exposure.as_array();

        for c in 0..num_communities {
            let mut hyp_total: f64 = 0.0;
            for g in 0..n_groups {
                let val = ge[[group, g]] + comp[[c, g]];
                hyp_row[g] = val;
                hyp_total += val;
            }
            if hyp_total < 1e-10 {
                hyp_total = 1e-10;
            }
            let mut dist_sq: f64 = 0.0;
            for g in 0..n_groups {
                let diff = (hyp_row[g] / hyp_total) - ideal[[group, g]];
                dist_sq += diff * diff;
            }
            distances[c] = dist_sq.sqrt();
        }

        if let Some(ref tc) = tc {
            let sizes = community_sizes.as_array();
            for c in 0..num_communities {
                if sizes[c] >= tc[c] {
                    distances[c] = f64::INFINITY;
                }
            }
        }

        let temperature: f64 = 1.0 - (node_idx as f64 / total_nodes as f64);
        let best_community: usize;

        if temperature > 0.05 {
            let valid: Vec<usize> = (0..num_communities)
                .filter(|&c| distances[c].is_finite())
                .collect();

            if valid.len() > 1 {
                let max_neg_d = valid
                    .iter()
                    .map(|&c| -distances[c] / (temperature + 1e-10))
                    .fold(f64::NEG_INFINITY, f64::max);

                let weights: Vec<f64> = valid
                    .iter()
                    .map(|&c| ((-distances[c] / (temperature + 1e-10)) - max_neg_d).exp())
                    .collect();

                let dist = WeightedIndex::new(&weights).unwrap();
                best_community = valid[dist.sample(&mut rng)];
            } else if valid.len() == 1 {
                best_community = valid[0];
            } else {
                best_community = distances
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
            }
        } else {
            best_community = distances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
        }

        assignments.push(best_community as i64);

        {
            let comp = community_composition.as_array();
            let mut ge = group_exposure.as_array_mut();
            for g in 0..n_groups {
                ge[[group, g]] += comp[[best_community, g]];
            }
            for g in 0..n_groups {
                if comp[[best_community, g]] > 0.0 {
                    ge[[g, group]] += 1.0;
                }
            }
        }

        {
            let mut comp = community_composition.as_array_mut();
            comp[[best_community, group]] += 1.0;
        }
        {
            let mut sizes = community_sizes.as_array_mut();
            sizes[best_community] += 1;
        }

        if (node_idx + 1) % 5000 == 0 {
            let pct = 100.0 * (node_idx + 1) as f64 / total_nodes as f64;
            println!(
                "Assigned {}/{} nodes ({:.1}%)",
                node_idx + 1,
                total_nodes,
                pct
            );
        }
    }

    let result = Array1::from(assignments);
    Ok(PyArray1::from_owned_array_bound(py, result))
}


/// Port of _run_edge_creation + establish_links while loop.
///
/// Processes all group pairs in one Rust call, maintaining the edge set
/// and adjacency list internally. Returns all new edges and final link counts.
#[pyfunction]
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None))]
fn run_edge_creation(
    // List of (src_id, dst_id, target_link_count) for each group pair
    group_pairs: Vec<(i64, i64, i64)>,
    // (src_id, dst_id) -> [community_ids] (may have duplicates for weighting)
    valid_communities_map: HashMap<(i64, i64), Vec<i64>>,
    // (src_id, dst_id) -> max link count
    maximum_num_links: HashMap<(i64, i64), i64>,
    // (community_id, group_id) -> [node_ids]
    communities_to_nodes: HashMap<(i64, i64), Vec<i64>>,
    // node_id -> group_id
    nodes_to_group: HashMap<i64, i64>,
    // Parameters
    fraction: f64,
    reciprocity_p: f64,
    transitivity_p: f64,
    pa_scope: String,
    number_of_communities: i64,
    bridge_probability: f64,
    // Optional pre-existing edges (for multiplex pre-seeding)
    pre_existing_edges: Option<Vec<(i64, i64)>>,
) -> PyResult<(Vec<(i64, i64)>, Vec<(i64, i64, i64)>)> {
    let mut rng = thread_rng();

    // Internal graph state
    let mut edges: HashSet<(i64, i64)> = HashSet::new();
    let mut adjacency: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut new_edges: Vec<(i64, i64)> = Vec::new();

    // Popularity pools: (community_id, group_id) -> [node_ids]
    let mut popularity_pool: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    // Link counters
    let mut existing_num_links: HashMap<(i64, i64), i64> = HashMap::new();
    for &(src, dst) in maximum_num_links.keys() {
        existing_num_links.insert((src, dst), 0);
    }

    // Initialize internal state from pre-existing edges (multiplex pre-seeding)
    if let Some(ref pre_edges) = pre_existing_edges {
        for &(s, d) in pre_edges {
            edges.insert((s, d));
            adjacency.entry(s).or_default().push(d);
            // Count toward link budget (do NOT add to new_edges — already in graph)
            let s_group = *nodes_to_group.get(&s).unwrap_or(&-1);
            let d_group = *nodes_to_group.get(&d).unwrap_or(&-1);
            if s_group >= 0 && d_group >= 0 {
                *existing_num_links.entry((s_group, d_group)).or_insert(0) += 1;
            }
        }
        if !pre_edges.is_empty() {
            println!("  Rust: initialized with {} pre-existing edges", pre_edges.len());
        }
    }

    // Src node list cache per (community_id, group_id)
    let mut src_node_cache: HashMap<(i64, i64), Vec<i64>> = HashMap::new();

    let total_pairs = group_pairs.len();

    for (pair_idx, (src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        let src_id = *src_id;
        let dst_id = *dst_id;
        let target_link_count = *target_link_count;

        if (pair_idx + 1) % 5000 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let possible_communities = match valid_communities_map.get(&(src_id, dst_id)) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };

        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);

        if num_links >= target_link_count {
            continue;
        }

        let max_attempts = target_link_count * 10;
        let mut attempts: i64 = 0;

        // Batch community selection
        let batch_size: usize = 10000;
        let pc_len = possible_communities.len();
        let mut community_batch: Vec<i64> = (0..batch_size)
            .map(|_| possible_communities[rng.gen_range(0..pc_len)])
            .collect();
        let mut batch_idx: usize = 0;

        while num_links < target_link_count && attempts < max_attempts {
            let community_id = community_batch[batch_idx];
            batch_idx += 1;

            if batch_idx >= batch_size {
                community_batch = (0..batch_size)
                    .map(|_| possible_communities[rng.gen_range(0..pc_len)])
                    .collect();
                batch_idx = 0;
            }

            // Get src nodes for this community
            let src_cache_key = (community_id, src_id);
            if !src_node_cache.contains_key(&src_cache_key) {
                let nodes = communities_to_nodes
                    .get(&src_cache_key)
                    .cloned()
                    .unwrap_or_default();
                src_node_cache.insert(src_cache_key, nodes);
            }
            let src_nodes = src_node_cache.get(&src_cache_key).unwrap();
            if src_nodes.is_empty() {
                attempts += 1;
                continue;
            }

            // Decide: bridge edge (dst from neighboring community) or normal
            let dst_community = if bridge_probability > 0.0
                && number_of_communities > 1
                && rng.gen::<f64>() < bridge_probability
            {
                let direction: i64 = if rng.gen::<bool>() { 1 } else { -1 };
                ((community_id + direction).rem_euclid(number_of_communities)) as i64
            } else {
                community_id
            };

            // Initialize popularity pool for (dst_community, dst_group)
            let pool_key = (dst_community, dst_id);
            if !popularity_pool.contains_key(&pool_key) {
                let dst_nodes = communities_to_nodes
                    .get(&pool_key)
                    .cloned()
                    .unwrap_or_default();
                if !dst_nodes.is_empty() {
                    let sample_size = ((dst_nodes.len() as f64) * fraction).ceil() as usize;
                    let sample_size = sample_size.min(dst_nodes.len());
                    let mut sampled = dst_nodes;
                    sampled.shuffle(&mut rng);
                    sampled.truncate(sample_size);
                    popularity_pool.insert(pool_key, sampled);
                } else {
                    popularity_pool.insert(pool_key, vec![]);
                }
            }

            let pool = popularity_pool.get(&pool_key).unwrap();
            if pool.is_empty() {
                attempts += 1;
                continue;
            }

            // Pick random src and dst
            let s = src_nodes[rng.gen_range(0..src_nodes.len())];
            let d = pool[rng.gen_range(0..pool.len())];

            if s != d && !edges.contains(&(s, d)) {
                edges.insert((s, d));
                adjacency.entry(s).or_default().push(d);
                new_edges.push((s, d));
                num_links += 1;
                existing_num_links.insert((src_id, dst_id), num_links);

                // Reciprocity
                if rng.gen::<f64>() < reciprocity_p {
                    let rev_existing = *existing_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    let rev_max = *maximum_num_links.get(&(dst_id, src_id)).unwrap_or(&0);
                    if rev_existing < rev_max && !edges.contains(&(d, s)) {
                        edges.insert((d, s));
                        adjacency.entry(d).or_default().push(s);
                        new_edges.push((d, s));
                        *existing_num_links.entry((dst_id, src_id)).or_insert(0) += 1;
                        if dst_id == src_id {
                            num_links += 1;
                            existing_num_links.insert((src_id, dst_id), num_links);
                        }
                    }
                }

                // Preferential attachment
                if rng.gen::<f64>() > fraction && fraction != 1.0 {
                    if pa_scope == "global" {
                        for comm_id in 0..number_of_communities {
                            if rng.gen::<f64>() < (1.0 / number_of_communities as f64) * fraction {
                                let global_key = (comm_id, dst_id);
                                if let Some(p) = popularity_pool.get_mut(&global_key) {
                                    p.push(d);
                                }
                            }
                        }
                    } else {
                        if rng.gen::<f64>() > fraction {
                            if let Some(p) = popularity_pool.get_mut(&pool_key) {
                                p.push(d);
                                // Also add a random node from the full dst community
                                if let Some(dst_community_nodes) = communities_to_nodes.get(&pool_key) {
                                    if !dst_community_nodes.is_empty() {
                                        let dst_random_community_node = dst_community_nodes[rng.gen_range(0..dst_community_nodes.len())];
                                        p.push(dst_random_community_node);
                                    }
                                }
                            }
                        }
                    }
                }

                // Transitivity
                if transitivity_p >= rng.gen::<f64>() {
                    let neighbors: Vec<i64> = adjacency
                        .get(&d)
                        .cloned()
                        .unwrap_or_default();
                    for n in neighbors {
                        if s == n {
                            continue;
                        }
                        let n_id = match nodes_to_group.get(&n) {
                            Some(&id) => id,
                            None => continue,
                        };
                        let pair = (src_id, n_id);
                        let max_l = match maximum_num_links.get(&pair) {
                            Some(&v) => v,
                            None => continue,
                        };
                        let existing = *existing_num_links.get(&pair).unwrap_or(&0);
                        if existing < max_l && !edges.contains(&(s, n)) {
                            edges.insert((s, n));
                            adjacency.entry(s).or_default().push(n);
                            new_edges.push((s, n));
                            *existing_num_links.entry(pair).or_insert(0) += 1;

                            if n_id == dst_id {
                                num_links += 1;
                                existing_num_links.insert((src_id, dst_id), num_links);
                            }

                            // Reciprocity for transitive edge
                            if rng.gen::<f64>() < reciprocity_p {
                                let rev_pair = (n_id, src_id);
                                let rev_existing =
                                    *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                let rev_max =
                                    *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                    edges.insert((n, s));
                                    adjacency.entry(n).or_default().push(s);
                                    new_edges.push((n, s));
                                    *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                    if n_id == src_id && src_id == dst_id {
                                        num_links += 1;
                                        existing_num_links
                                            .insert((src_id, dst_id), num_links);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            attempts += 1;
        }
    }

    // Convert existing_num_links to flat triples
    let links_out: Vec<(i64, i64, i64)> = existing_num_links
        .into_iter()
        .map(|((s, d), c)| (s, d, c))
        .collect();

    Ok((new_edges, links_out))
}


/// Soft-penalty cost for remaining edge budget.
/// Overshoot (negative remainder) is penalised 10×.
#[inline(always)]
fn comm_cost(x: f64) -> f64 {
    if x >= 0.0 { x * x } else { 10.0 * x * x }
}

/// SA softmax selection over `(community_id, distance)` candidates.
fn sa_select(candidates: &[(usize, f64)], temperature: f64, rng: &mut impl Rng) -> usize {
    if temperature > 0.05 {
        let valid: Vec<_> = candidates.iter().filter(|&&(_, d)| d.is_finite()).collect();
        match valid.len() {
            0 => candidates.last().map(|&(i, _)| i).unwrap_or(0),
            1 => valid[0].0,
            _ => {
                let t = temperature + 1e-10;
                let max_neg = valid.iter().map(|&&(_, d)| -d / t).fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = valid.iter()
                    .map(|&&(_, d)| ((-d / t) - max_neg).exp())
                    .collect();
                valid[WeightedIndex::new(&weights).unwrap().sample(rng)].0
            }
        }
    } else {
        candidates.iter()
            .filter(|&&(_, d)| d.is_finite())
            .min_by(|&&(_, a), &&(_, b)| a.partial_cmp(&b).unwrap())
            .map(|&(i, _)| i)
            .unwrap_or_else(|| candidates.last().map(|&(i, _)| i).unwrap_or(0))
    }
}


/// Sparse-SA community assignment based on edge-budget fulfillment.
///
/// Each node joins the community that best fulfills the edge-budget targets
/// for its group.  Only communities that share at least one budget-neighbour
/// with the current node's group are evaluated ("warm set"); all others share
/// the same baseline distance and are never preferred over warm ones.
///
/// Memory: O(communities × avg_groups_per_community + budget_pairs)
/// instead of the previous O(communities × n_groups + n_groups²).
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, budget, n_groups, initial_num_communities, target_counts=None, total_nodes=0, new_comm_penalty=3.0, initial_comp=None))]
fn process_nodes_capacity<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    initial_num_communities: usize,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
    new_comm_penalty: f64,
    initial_comp: Option<HashMap<usize, HashMap<usize, i64>>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (all_nodes, initial_num_communities);
    let node_groups = node_groups.as_array();
    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());
    let mut rng = thread_rng();

    // ── Sparse budget neighbour lists ─────────────────────────────────────
    // nbrs_row[g] = [(h, target[g→h])]  nbrs_col[g] = [(h, target[h→g]), h≠g]
    let mut nbrs_row: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n_groups];
    let mut nbrs_col: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n_groups];
    let mut budget_nbrs: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (&(sg, dg), &val) in &budget {
        if val <= 0 { continue; }
        let sg = sg as usize;
        let dg = dg as usize;
        if sg < n_groups && dg < n_groups {
            nbrs_row[sg].push((dg, val));
            budget_nbrs[sg].push(dg);
            budget_nbrs[dg].push(sg);
            if sg != dg { nbrs_col[dg].push((sg, val)); }
        }
    }
    for v in &mut budget_nbrs { v.sort_unstable(); v.dedup(); v.shrink_to_fit(); }

    // -- Community state (sparse) --
    let seed_count = initial_comp.as_ref().map(|ic| ic.len()).unwrap_or(0);
    let mut comp: Vec<HashMap<usize, i64>>  = Vec::with_capacity(seed_count.max(16));
    let mut comm_sizes: Vec<i64>            = Vec::with_capacity(seed_count.max(16));
    let mut num_active: usize = 0;
    let mut comms_with_group: HashMap<usize, Vec<usize>> = HashMap::new();

    if let Some(ref ic) = initial_comp {
        while comp.len() < ic.len() { comp.push(HashMap::new()); comm_sizes.push(0); }
        for (&cid, dict) in ic {
            if cid < comp.len() {
                for (&gid, &cnt) in dict {
                    if gid < n_groups && cnt > 0 {
                        *comp[cid].entry(gid).or_insert(0) += cnt;
                        comm_sizes[cid] += cnt;
                        comms_with_group.entry(gid).or_default().push(cid);
                    }
                }
            }
        }
        num_active = ic.len();
    }

    // -- acc_sym[g][h]: accumulated edge-opportunities, sparse, O(budget_pairs) --
    let mut acc_sym: HashMap<usize, HashMap<usize, i64>> = HashMap::new();

    const MAX_EVAL: usize = 500;
    let mut warm:       Vec<usize>        = Vec::new();
    let mut seen:       HashSet<usize>    = HashSet::new();
    let mut candidates: Vec<(usize, f64)> = Vec::new();
    let mut assignments: Vec<i64>         = Vec::with_capacity(total_nodes);

    for node_idx in 0..total_nodes {
        let g = node_groups[node_idx] as usize;

        // -- Baseline: cost of g joining an empty community --
        let mut base_sq = 0.0f64;
        {
            let g_acc = acc_sym.get(&g);
            for &(h, tv) in &nbrs_row[g] {
                let av = g_acc.and_then(|m| m.get(&h)).copied().unwrap_or(0) as f64;
                base_sq += comm_cost(tv as f64 - av);
            }
            for &(h, tv) in &nbrs_col[g] {
                let av = g_acc.and_then(|m| m.get(&h)).copied().unwrap_or(0) as f64;
                base_sq += comm_cost(tv as f64 - av);
            }
        }

        // -- Warm set --
        warm.clear();
        seen.clear();
        for &(h, _) in nbrs_row[g].iter().chain(nbrs_col[g].iter()) {
            if let Some(cs) = comms_with_group.get(&h) {
                for &c in cs {
                    if c < num_active && seen.insert(c) { warm.push(c); }
                }
            }
        }
        if warm.len() > MAX_EVAL {
            warm.partial_shuffle(&mut rng, MAX_EVAL);
            warm.truncate(MAX_EVAL);
        }

        // -- Candidates: warm communities + new community --
        candidates.clear();
        {
            let g_acc = acc_sym.get(&g);
            for &c in &warm {
                if let Some(ref tc_arr) = tc {
                    if c < tc_arr.len() && comm_sizes[c] as i32 >= tc_arr[c] { continue; }
                }
                let c_comp = &comp[c];
                let mut delta = 0.0f64;
                for &(h, tv) in &nbrs_row[g] {
                    let cnt = c_comp.get(&h).copied().unwrap_or(0);
                    if cnt == 0 { continue; }
                    let av = g_acc.and_then(|m| m.get(&h)).copied().unwrap_or(0) as f64;
                    let tv = tv as f64;
                    delta += if h == g {
                        comm_cost(tv - av - 2.0 * cnt as f64) - comm_cost(tv - av)
                    } else {
                        comm_cost(tv - av - cnt as f64) - comm_cost(tv - av)
                    };
                }
                for &(h, tv) in &nbrs_col[g] {
                    let cnt = c_comp.get(&h).copied().unwrap_or(0);
                    if cnt == 0 { continue; }
                    let av = g_acc.and_then(|m| m.get(&h)).copied().unwrap_or(0) as f64;
                    delta += comm_cost(tv as f64 - av - cnt as f64) - comm_cost(tv as f64 - av);
                }
                candidates.push((c, (base_sq + delta).max(0.0).sqrt()));
            }
        }
        // Only offer a new community when allowed. Skipping this when
        // new_comm_penalty is infinite avoids the 0.0 * inf = NaN trap
        // that would otherwise slip through is_finite() checks.
        if new_comm_penalty.is_finite() {
            candidates.push((num_active, base_sq.sqrt() * new_comm_penalty));
        }

        // -- SA selection --
        let temperature = 1.0 - (node_idx as f64 / total_nodes as f64);
        let chosen = if candidates.is_empty() {
            // Group has no warm neighbours and new communities are disallowed:
            // assign uniformly to any existing community.
            rng.gen_range(0..num_active.max(1))
        } else {
            sa_select(&candidates, temperature, &mut rng)
        };

        let best = if chosen >= num_active {
            // Only reachable when new_comm_penalty is finite (candidate was added above).
            comp.push(HashMap::new());
            comm_sizes.push(0);
            num_active += 1;
            chosen
        } else {
            chosen
        };

        // -- Update acc_sym --
        {
            let diag_cnt = comp[best].get(&g).copied().unwrap_or(0);
            let off: Vec<(usize, i64)> = budget_nbrs[g].iter()
                .filter(|&&h| h != g)
                .filter_map(|&h| comp[best].get(&h).map(|&cnt| (h, cnt)))
                .filter(|&(_, cnt)| cnt > 0)
                .collect();
            if diag_cnt > 0 {
                *acc_sym.entry(g).or_default().entry(g).or_insert(0) += 2 * diag_cnt;
            }
            for (h, cnt) in off {
                *acc_sym.entry(g).or_default().entry(h).or_insert(0) += cnt;
                *acc_sym.entry(h).or_default().entry(g).or_insert(0) += cnt;
            }
        }

        // -- Update comp + inverted index --
        let prev = comp[best].get(&g).copied().unwrap_or(0);
        *comp[best].entry(g).or_insert(0) += 1;
        comm_sizes[best] += 1;
        if prev == 0 { comms_with_group.entry(g).or_default().push(best); }

        assignments.push(best as i64);

        if (node_idx + 1) % 5000 == 0 {
            println!(
                "Capacity SA: {}/{} ({:.1}%), {} communities",
                node_idx + 1, total_nodes,
                100.0 * (node_idx + 1) as f64 / total_nodes as f64,
                num_active
            );
        }
    }

    println!("Capacity SA complete: {} nodes -> {} communities", total_nodes, num_active);
    Ok(PyArray1::from_owned_array_bound(py, Array1::from(assignments)))
}

/// Sparse simulated annealing for large group counts.
///
/// Replaces the two O(n_groups²) flat arrays in `process_nodes_capacity` with
/// sparse HashMaps, reducing memory from ~39 GB to O(nnz + n_nodes).
///
/// For K average non-zero target neighbours per group and C active communities
/// the inner loop is O(K·C) per node instead of O(n_groups·C).
///
/// Drop-in compatible: same signature and return type as `process_nodes_capacity`.
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, budget, n_groups, initial_num_communities, target_counts=None, total_nodes=0, new_comm_penalty=3.0, initial_comp=None))]
fn process_nodes_capacity_sparse<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    initial_num_communities: usize,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
    new_comm_penalty: f64,
    initial_comp: Option<HashMap<usize, HashMap<usize, i64>>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = all_nodes; // caller already shuffled all_nodes/node_groups in Python
    let _ = initial_num_communities; // no longer used to pre-allocate (memory)
    let node_groups_arr = node_groups.as_array();
    let tc: Option<Array1<i32>> = target_counts.map(|t| t.as_array().to_owned());

    let mut rng = thread_rng();

    // ── Sparse neighbour lists built from budget (O(nnz)) ─────────────────
    // budget_nbrs is built directly as Vec<Vec<usize>> (no temp HashSet) and
    // de-duplicated with sort + dedup. Saves ~48·n_groups bytes of peak
    // memory vs. the previous HashSet-then-convert approach.
    let mut nbrs_row: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n_groups];
    let mut nbrs_col: Vec<Vec<(usize, i64)>> = vec![Vec::new(); n_groups];
    let mut budget_nbrs: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (&(sg, dg), &val) in &budget {
        if val <= 0 { continue; }
        let sg = sg as usize;
        let dg = dg as usize;
        if sg < n_groups && dg < n_groups {
            nbrs_row[sg].push((dg, val));
            budget_nbrs[sg].push(dg);
            budget_nbrs[dg].push(sg);
            if sg != dg {
                nbrs_col[dg].push((sg, val));
            }
        }
    }
    for v in &mut budget_nbrs {
        if v.len() > 1 {
            v.sort_unstable();
            v.dedup();
        }
        v.shrink_to_fit();
    }

    // ── Community composition: comp_by_comm[c][h] = count ────────────────
    // Grow LAZILY — never pre-allocate to a "cap" hint. An empty HashMap is
    // ~48 bytes; pre-allocating tens of millions of them is multi-GB.
    let initial_seed_count = initial_comp.as_ref().map(|ic| ic.len()).unwrap_or(0);
    let initial_alloc = initial_seed_count.max(64);
    let mut comp_by_comm: Vec<HashMap<usize, i64>> = Vec::with_capacity(initial_alloc);
    let mut comm_sizes: Vec<i64> = Vec::with_capacity(initial_alloc);
    let mut num_active: usize = 0;

    // ── INVERTED INDEX (sparse) ───────────────────────────────────────────
    // comms_with_group[h] = set of community ids currently containing ≥1
    // member of group h. Used to compute the WARM SET per node.
    //
    // HashMap (not Vec<HashSet>): memory scales with the number of
    // *populated* groups, not n_groups. For huge n_groups this is the
    // difference between MB and GB.
    let mut comms_with_group: HashMap<usize, HashSet<usize>> = HashMap::new();

    // Initialize from pre-seeded communities (each seed alone in its community)
    if let Some(ref ic) = initial_comp {
        let needed = ic.len();
        while comp_by_comm.len() < needed {
            comp_by_comm.push(HashMap::new());
            comm_sizes.push(0);
        }
        for (&comm_id, comp_dict) in ic {
            if comm_id < needed {
                for (&group_id, &count) in comp_dict {
                    if group_id < n_groups && count > 0 {
                        *comp_by_comm[comm_id].entry(group_id).or_insert(0) += count;
                        comm_sizes[comm_id] += count;
                        comms_with_group
                            .entry(group_id)
                            .or_insert_with(HashSet::new)
                            .insert(comm_id);
                    }
                }
            }
        }
        num_active = needed;
        // acc_sym stays zero — seeds are each in their own community, no pairs yet
    }

    // ── Accumulated co-location counts: acc_sym[g][h] (symmetric) ────────
    // NOTE: still a Vec<HashMap> of size n_groups. If memory is still a
    // problem at very large n_groups, sparsify this to
    // HashMap<usize, HashMap<usize, i64>> with the same pattern as
    // comms_with_group above.
    let mut acc_sym: Vec<HashMap<usize, i64>> = (0..n_groups).map(|_| HashMap::new()).collect();

    // Reusable buffers — grow on demand
    let mut distances: Vec<f64> = vec![0.0; initial_alloc + 1];
    let mut warm: HashSet<usize> = HashSet::new();
    let empty_set: HashSet<usize> = HashSet::new(); // sentinel for missed lookups

    #[inline(always)]
    fn cost(x: f64) -> f64 {
        const OVERSHOOT: f64 = 10.0;
        if x >= 0.0 { x * x } else { OVERSHOOT * x * x }
    }

    let mut assignments: Vec<i64> = Vec::with_capacity(total_nodes);

    for node_idx in 0..total_nodes {
        let g = node_groups_arr[node_idx] as usize;

        // ── Base distance: placing g in a new empty community ─────────────
        let base: f64 = {
            let g_acc = &acc_sym[g];
            let mut d = 0.0f64;
            for &(h, tv) in &nbrs_row[g] {
                let av = *g_acc.get(&h).unwrap_or(&0) as f64;
                d += cost(tv as f64 - av);
            }
            for &(h, tv) in &nbrs_col[g] {
                let av = *g_acc.get(&h).unwrap_or(&0) as f64;
                d += cost(tv as f64 - av);
            }
            d
        };
        let new_dist = new_comm_penalty * base.sqrt();

        // Grow distances buffer if needed (amortized O(1))
        if distances.len() <= num_active {
            let new_len = (num_active + 16).max(distances.len() * 2);
            distances.resize(new_len, 0.0);
        }

        // ── Build WARM SET: communities sharing a budget-neighbour with g ──
        warm.clear();
        for &(h, _) in &nbrs_row[g] {
            for &c in comms_with_group.get(&h).unwrap_or(&empty_set).iter() {
                warm.insert(c);
            }
        }
        for &(h, _) in &nbrs_col[g] {
            for &c in comms_with_group.get(&h).unwrap_or(&empty_set).iter() {
                warm.insert(c);
            }
        }

        // ── Initialise distances: cold default = base, capped = ∞ ─────────
        if let Some(ref tc_ref) = tc {
            for c in 0..num_active {
                let capped = c < tc_ref.len() && comm_sizes[c] as i32 >= tc_ref[c];
                distances[c] = if capped { f64::INFINITY } else { base };
            }
        } else {
            for c in 0..num_active {
                distances[c] = base;
            }
        }

        // ── Override warm communities with their actual distance ──────────
        for &c in warm.iter() {
            if c >= num_active { continue; }
            if !distances[c].is_finite() { continue; } // capped; skip

            let c_comp = &comp_by_comm[c];
            let g_acc  = &acc_sym[g];
            let mut delta = 0.0f64;

            for &(h, tv) in &nbrs_row[g] {
                let count_h = *c_comp.get(&h).unwrap_or(&0);
                if count_h == 0 { continue; }
                let av  = *g_acc.get(&h).unwrap_or(&0) as f64;
                let tv  = tv as f64;
                let old = cost(tv - av);
                if h == g {
                    delta += cost(tv - av - 2.0 * count_h as f64) - old;
                } else {
                    delta += cost(tv - av - count_h as f64) - old;
                }
            }
            for &(h, tv) in &nbrs_col[g] {
                let count_h = *c_comp.get(&h).unwrap_or(&0);
                if count_h == 0 { continue; }
                let av  = *g_acc.get(&h).unwrap_or(&0) as f64;
                let tv  = tv as f64;
                delta += cost(tv - av - count_h as f64) - cost(tv - av);
            }

            distances[c] = base + delta;
        }
        distances[num_active] = new_dist;

        // ── SA temperature-based selection ────────────────────────────────
        let temperature = 1.0 - (node_idx as f64 / total_nodes as f64);

        let chosen = if temperature > 0.05 {
            let valid: Vec<usize> = (0..=num_active)
                .filter(|&c| distances[c].is_finite())
                .collect();

            if valid.len() > 1 {
                let max_neg = valid.iter()
                    .map(|&c| -distances[c] / (temperature + 1e-10))
                    .fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = valid.iter()
                    .map(|&c| ((-distances[c] / (temperature + 1e-10)) - max_neg).exp())
                    .collect();
                let dist = WeightedIndex::new(&weights).unwrap();
                valid[dist.sample(&mut rng)]
            } else if valid.len() == 1 {
                valid[0]
            } else {
                num_active  // fallback: new community
            }
        } else {
            (0..=num_active)
                .filter(|&c| distances[c].is_finite())
                .min_by(|&a, &b| distances[a].partial_cmp(&distances[b]).unwrap())
                .unwrap_or(num_active)
        };

        // ── Allocate new community on demand (no pre-allocation to cap) ───
        let best_comm = if chosen >= num_active {
            let idx = num_active;
            if comp_by_comm.len() <= idx {
                comp_by_comm.push(HashMap::new());
                comm_sizes.push(0);
            }
            num_active += 1;
            idx
        } else {
            chosen
        };

        // ── Update acc_sym: iterate budget_nbrs[g], not comp_by_comm.
        //    O(deg(g)) instead of O(|community|).
        {
            let comp = &comp_by_comm[best_comm];
            let mut diag_add: i64 = 0;
            let mut updates: Vec<(usize, i64)> = Vec::with_capacity(budget_nbrs[g].len());

            if let Some(&cnt) = comp.get(&g) {
                if cnt > 0 { diag_add = 2 * cnt; }
            }
            for &h in &budget_nbrs[g] {
                if h == g { continue; }
                if let Some(&cnt) = comp.get(&h) {
                    if cnt > 0 { updates.push((h, cnt)); }
                }
            }

            if diag_add != 0 {
                *acc_sym[g].entry(g).or_insert(0) += diag_add;
            }
            for (h, cnt) in updates {
                *acc_sym[g].entry(h).or_insert(0) += cnt;
                *acc_sym[h].entry(g).or_insert(0) += cnt;
            }
        }

        // ── Update composition + inverted index ───────────────────────────
        let was_present = comp_by_comm[best_comm].get(&g).copied().unwrap_or(0) > 0;
        *comp_by_comm[best_comm].entry(g).or_insert(0) += 1;
        comm_sizes[best_comm] += 1;
        if !was_present {
            comms_with_group
                .entry(g)
                .or_insert_with(HashSet::new)
                .insert(best_comm);
        }

        assignments.push(best_comm as i64);

        if (node_idx + 1) % 5000 == 0 {
            println!(
                "Sparse SA: {}/{} nodes ({:.1}%), {} active communities",
                node_idx + 1,
                total_nodes,
                100.0 * (node_idx + 1) as f64 / total_nodes as f64,
                num_active
            );
        }
    }

    println!(
        "Sparse SA complete: {} nodes -> {} active communities",
        total_nodes, num_active
    );

    let result = Array1::from(assignments);
    Ok(PyArray1::from_owned_array_bound(py, result))
}

/// Fast O(N) community assignment — no SA, uniform group distribution.
///
/// For each group g with p_g nodes, distributes them as evenly as possible
/// across all K communities (floor/ceil split, randomly shuffled per group).
/// Runs in O(N + n_groups × K) time with no HashMap lookups in the hot path.
///
/// Quality trade-off: all communities end up with identical demographic
/// composition (± 1 node per group). Budget structure is ignored — the SA
/// variants optimise jointly; this does not.
#[pyfunction]
#[pyo3(signature = (all_nodes, node_groups, budget, n_groups, initial_num_communities, target_counts=None, total_nodes=0, new_comm_penalty=3.0, initial_comp=None))]
fn process_nodes_capacity_fast<'py>(
    py: Python<'py>,
    all_nodes: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    initial_num_communities: usize,
    target_counts: Option<PyReadonlyArray1<'py, i32>>,
    total_nodes: usize,
    new_comm_penalty: f64,
    initial_comp: Option<HashMap<usize, HashMap<usize, i64>>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let _ = (all_nodes, budget, new_comm_penalty, initial_comp);
    let node_groups = node_groups.as_array();
    let k = initial_num_communities.max(1);
    let mut rng = thread_rng();

    let tc: Option<Vec<i32>> = target_counts.map(|t| t.as_array().to_owned().to_vec());

    // Step 1: count nodes per group in one pass
    let mut group_counts = vec![0usize; n_groups];
    for i in 0..total_nodes {
        let g = node_groups[i] as usize;
        if g < n_groups {
            group_counts[g] += 1;
        }
    }

    // Step 2: for each group build a shuffled list of community assignments
    let mut group_lists: Vec<Vec<usize>> = (0..n_groups)
        .map(|g| {
            let p = group_counts[g];
            if p == 0 {
                return Vec::new();
            }
            let mut list = Vec::with_capacity(p);

            if let Some(ref tc_arr) = tc {
                // Proportional to target community sizes
                let total_target: i64 = tc_arr.iter().map(|&x| x as i64).sum();
                let denom = if total_target > 0 { total_target as f64 } else { k as f64 };
                let mut assigned = 0usize;
                for c in 0..k {
                    let share = if c < k - 1 {
                        let raw = (p as f64 * tc_arr[c] as f64 / denom).round() as usize;
                        raw.min(p - assigned)
                    } else {
                        p - assigned
                    };
                    for _ in 0..share {
                        list.push(c);
                    }
                    assigned += share;
                }
            } else {
                // Uniform: shuffle community order so different groups vary
                let mut comm_order: Vec<usize> = (0..k).collect();
                comm_order.shuffle(&mut rng);
                let base = p / k;
                let remainder = p % k;
                for (i, &c) in comm_order.iter().enumerate() {
                    let count = base + if i < remainder { 1 } else { 0 };
                    for _ in 0..count {
                        list.push(c);
                    }
                }
            }

            list.shuffle(&mut rng);
            list
        })
        .collect();

    // Step 3: assign each node by popping from its group's list
    let mut group_cursors = vec![0usize; n_groups];
    let mut assignments = Vec::with_capacity(total_nodes);

    for i in 0..total_nodes {
        let g = node_groups[i] as usize;
        let comm = if g < n_groups && group_cursors[g] < group_lists[g].len() {
            let c = group_lists[g][group_cursors[g]];
            group_cursors[g] += 1;
            c
        } else {
            rng.gen_range(0..k)
        };
        assignments.push(comm as i64);
    }

    println!(
        "Fast assignment complete: {} nodes -> {} communities",
        total_nodes, k
    );
    Ok(PyArray1::from_owned_array_bound(py, Array1::from(assignments)))
}

/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity_fast, m)?)?;
    Ok(())
}
