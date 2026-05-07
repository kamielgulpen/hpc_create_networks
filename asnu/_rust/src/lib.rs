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
#[pyo3(signature = (group_pairs, valid_communities_map, maximum_num_links, communities_to_nodes, nodes_to_group, fraction, reciprocity_p, transitivity_p, pa_scope, number_of_communities, bridge_probability=0.0, pre_existing_edges=None, node_coordinates=None))]
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
    // Optional node coordinates for Phase B spatial ring search
    node_coordinates: Option<HashMap<i64, f64>>,
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

    // Phase B: src nodes sorted by coordinate, dst communities sorted by centroid.
    // Ring search finds nearest communities then picks a random node from each —
    // spreading degree load across all nodes rather than targeting edge-nearest ones.
    let mut group_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    let mut group_comm_sorted: HashMap<i64, Vec<(f64, i64)>> = HashMap::new();
    if let Some(ref nc) = node_coordinates {
        let mut group_all_nodes: HashMap<i64, Vec<i64>> = HashMap::new();
        for (&(_, gid), nodes) in &communities_to_nodes {
            group_all_nodes.entry(gid).or_default().extend(nodes.iter().copied());
        }
        for (gid, nodes) in &group_all_nodes {
            let mut sorted: Vec<(f64, i64)> = nodes.iter()
                .map(|&n| (*nc.get(&n).unwrap_or(&0.5), n))
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            group_sorted.insert(*gid, sorted);
        }
        // Community centroids: average coordinate of nodes in (comm, group)
        for (&(comm_id, gid), nodes) in &communities_to_nodes {
            if nodes.is_empty() { continue; }
            let centroid: f64 = nodes.iter()
                .map(|&n| *nc.get(&n).unwrap_or(&0.5))
                .sum::<f64>() / nodes.len() as f64;
            group_comm_sorted.entry(gid).or_default().push((centroid, comm_id));
        }
        for sorted in group_comm_sorted.values_mut() {
            sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        println!("  Phase B: built sorted arrays for {} groups", group_sorted.len());
    }

    let total_pairs = group_pairs.len();

    for (pair_idx, (src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        let src_id = *src_id;
        let dst_id = *dst_id;
        let target_link_count = *target_link_count;

        if (pair_idx + 1) % 5000 == 0 || pair_idx == 0 || pair_idx == total_pairs - 1 {
            println!("Processing pair {} of {}", pair_idx + 1, total_pairs);
        }

        let possible_communities = valid_communities_map
            .get(&(src_id, dst_id))
            .filter(|v| !v.is_empty());

        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);

        if num_links >= target_link_count {
            continue;
        }

        // ── Phase A: community-based edge creation ────────────────────────────
        // Communities are iterated sequentially (shuffled once per pair) so each
        // community exhausts a proportional quota before moving to the next.
        // This concentrates edges within communities, raising transitivity.
        if let Some(communities) = possible_communities {
            let mut comm_order: Vec<i64> = {
                let mut seen = std::collections::HashSet::new();
                communities.iter().filter(|&&c| seen.insert(c)).cloned().collect()
            };
            comm_order.shuffle(&mut rng);
            let n_comms = comm_order.len();

            let max_passes: i64 = 3;
            let mut pass: i64 = 0;

            'outer: while num_links < target_link_count && pass < max_passes {
                pass += 1;
                for &community_id in &comm_order {
                    if num_links >= target_link_count { break 'outer; }

                    let remaining = (target_link_count - num_links) as usize;
                    let quota = ((remaining + n_comms - 1) / n_comms).max(1);

                    // Get src nodes for this community
                    let src_cache_key = (community_id, src_id);
                    if !src_node_cache.contains_key(&src_cache_key) {
                        let nodes = communities_to_nodes
                            .get(&src_cache_key)
                            .cloned()
                            .unwrap_or_default();
                        src_node_cache.insert(src_cache_key, nodes);
                    }
                    if src_node_cache.get(&src_cache_key).unwrap().is_empty() {
                        continue;
                    }

                    // Bridge or normal dst community
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
                    if popularity_pool.get(&pool_key).unwrap().is_empty() {
                        continue;
                    }

                    // Create up to quota edges within this community
                    let mut created = 0usize;
                    let mut local_attempts = 0usize;
                    let max_local = quota * 3;

                    while created < quota && local_attempts < max_local && num_links < target_link_count {
                        local_attempts += 1;

                        let s = {
                            let src_nodes = src_node_cache.get(&src_cache_key).unwrap();
                            src_nodes[rng.gen_range(0..src_nodes.len())]
                        };
                        let d = {
                            let pool = popularity_pool.get(&pool_key).unwrap();
                            pool[rng.gen_range(0..pool.len())]
                        };

                        if s != d && !edges.contains(&(s, d)) {
                            edges.insert((s, d));
                            adjacency.entry(s).or_default().push(d);
                            new_edges.push((s, d));
                            num_links += 1;
                            existing_num_links.insert((src_id, dst_id), num_links);
                            created += 1;

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
                                            if let Some(dst_community_nodes) = communities_to_nodes.get(&pool_key) {
                                                if !dst_community_nodes.is_empty() {
                                                    let r = rng.gen_range(0..dst_community_nodes.len());
                                                    p.push(dst_community_nodes[r]);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Transitivity
                            if transitivity_p >= rng.gen::<f64>() {
                                let neighbors: Vec<i64> = adjacency.get(&d).cloned().unwrap_or_default();
                                for n in neighbors {
                                    if s == n { continue; }
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
                                            let rev_existing = *existing_num_links.get(&rev_pair).unwrap_or(&0);
                                            let rev_max = *maximum_num_links.get(&rev_pair).unwrap_or(&0);
                                            if !edges.contains(&(n, s)) && rev_existing < rev_max {
                                                edges.insert((n, s));
                                                adjacency.entry(n).or_default().push(s);
                                                new_edges.push((n, s));
                                                *existing_num_links.entry(rev_pair).or_insert(0) += 1;
                                                if n_id == src_id && src_id == dst_id {
                                                    num_links += 1;
                                                    existing_num_links.insert((src_id, dst_id), num_links);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // end Phase A
    } // end pair loop

    // ── Phase B: spatial ring search for remaining budget ────────────────────
    // For each pair still under budget, finds nearest dst communities by
    // centroid and picks a random node — fills cross-block pairs left by A/B.
    for (pair_idx, &(src_id, dst_id, target_link_count)) in group_pairs.iter().enumerate() {
        if (pair_idx + 1) % 5000 == 0 {
            println!("Phase B: pair {} of {}", pair_idx + 1, total_pairs);
        }
        let mut num_links = *existing_num_links.get(&(src_id, dst_id)).unwrap_or(&0);
        if num_links >= target_link_count { continue; }

        if let (Some(src_sorted), Some(dst_comm_sorted)) =
            (group_sorted.get(&src_id), group_comm_sorted.get(&dst_id))
        {
            const PHASE_C_COMM_WINDOW: usize = 200;
            let n_dst_comm = dst_comm_sorted.len();
            let win = PHASE_C_COMM_WINDOW.min(n_dst_comm);
            let n_src = src_sorted.len();

            if n_src > 0 && win > 0 {
                let mut src_indices: Vec<usize> = (0..n_src).collect();
                loop {
                    if num_links >= target_link_count { break; }
                    src_indices.shuffle(&mut rng);
                    let remaining = (target_link_count - num_links) as usize;
                    let edges_per_src = ((remaining + n_src - 1) / n_src).max(1).min(win);
                    let mut made_progress = false;

                    for &si in &src_indices {
                        if num_links >= target_link_count { break; }
                        let (theta_s, s) = src_sorted[si];
                        let center = dst_comm_sorted.partition_point(|&(c, _)| c < theta_s);
                        let mut found = 0usize;

                        'delta: for delta in 0..win {
                            if found >= edges_per_src { break 'delta; }
                            let j1 = (center + delta) % n_dst_comm;
                            let j2 = (center + n_dst_comm - delta - 1) % n_dst_comm;
                            for &j in &[j1, j2] {
                                if found >= edges_per_src || num_links >= target_link_count { break; }
                                let (_, comm_id) = dst_comm_sorted[j];
                                let pool_key = (comm_id, dst_id);
                                if let Some(dst_nodes) = communities_to_nodes.get(&pool_key) {
                                    if !dst_nodes.is_empty() {
                                        let d = dst_nodes[rng.gen_range(0..dst_nodes.len())];
                                        if s != d && !edges.contains(&(s, d)) {
                                            edges.insert((s, d));
                                            adjacency.entry(s).or_default().push(d);
                                            new_edges.push((s, d));
                                            *existing_num_links.entry((src_id, dst_id)).or_insert(0) += 1;
                                            num_links += 1;
                                            found += 1;
                                            made_progress = true;

                                            // Reciprocity (same logic as Phase A)
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
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if !made_progress { break; }
                }
            }
        }
    } // end Phase B

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


// ============================================================================
// Swap-based community refinement — adaptive sparse/dense storage
// ============================================================================
//
// At a glance:
//   - For small n_communities × n_groups: dense Vec<Vec<i64>> for comp (fast)
//   - For large: sparse HashMap with FxHash for cache-efficient lookups
//   - Biased proposals: 80% of swaps target over/under-budget cells
//   - Single-pass delta + achieved update over `comm_groups[c1] ∪ comm_groups[c2]`
//
// Choose the storage automatically based on a memory budget. The hot-path code
// is shared — only the comp accessor differs, and that's wrapped in an enum
// the compiler will monomorphise away on the dense branch.
// ============================================================================


/// 8 bytes per i64 entry. Allow up to ~1 GB before falling back to sparse.
const DENSE_MEMORY_BUDGET_BYTES: usize = 1_000_000_000;

/// Hot-path composition storage. Two backends share an interface so the
/// inner loop can be written once.
enum Comp {
    Dense(Vec<Vec<i64>>),               // [c][g] -> count
    Sparse(Vec<HashMap<usize, i64>>),   // [c] -> {g: count}
}

impl Comp {
    #[inline(always)]
    fn get(&self, c: usize, g: usize) -> i64 {
        match self {
            Comp::Dense(v) => v[c][g],
            Comp::Sparse(v) => *v[c].get(&g).unwrap_or(&0),
        }
    }

    #[inline(always)]
    fn set(&mut self, c: usize, g: usize, val: i64) {
        match self {
            Comp::Dense(v) => v[c][g] = val,
            Comp::Sparse(v) => {
                if val == 0 {
                    v[c].remove(&g);
                } else {
                    v[c].insert(g, val);
                }
            }
        }
    }

    #[inline(always)]
    fn add(&mut self, c: usize, g: usize, delta: i64) {
        match self {
            Comp::Dense(v) => v[c][g] += delta,
            Comp::Sparse(v) => {
                let entry = v[c].entry(g).or_insert(0);
                *entry += delta;
                if *entry == 0 { v[c].remove(&g); }
            }
        }
    }
}

/// Swap-based refinement of community assignments to better match the edge budget.
///
/// Designed to run AFTER a structural initialiser like `process_nodes_capacity_fast`.
/// Biased proposals target communities where some group is currently over- or
/// under-represented relative to the budget.
///
/// Loss: Σ_{g,h} w(g,h) · (achieved[g,h] − budget[g,h])²
///   where w = OVERSHOOT_PENALTY when achieved > budget, else 1.0.
/// achieved[g,h] = Σ_c C[c,g] · C[c,h]
///
/// Storage adapts to size: dense Vec<Vec<i64>> when n_communities × n_groups
/// fits in the memory budget, sparse HashMap otherwise.
#[pyfunction]
#[pyo3(signature = (
    assignments,
    node_groups,
    budget,
    n_groups,
    n_communities,
    n_iterations = 100_000,
    overshoot_penalty = 10.0,
    temperature_start = 1.0,
    seed = 42,
    biased_fraction = 0.8,
))]
fn refine_communities_swap<'py>(
    py: Python<'py>,
    assignments: PyReadonlyArray1<'py, i64>,
    node_groups: PyReadonlyArray1<'py, i64>,
    budget: HashMap<(i64, i64), i64>,
    n_groups: usize,
    n_communities: usize,
    n_iterations: usize,
    overshoot_penalty: f64,
    temperature_start: f64,
    seed: u64,
    biased_fraction: f64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let assignments_in = assignments.as_array();
    let node_groups_arr = node_groups.as_array();
    let total_nodes = assignments_in.len();

    let mut rng = StdRng::seed_from_u64(seed);

    // ── Decide storage backend based on memory budget ────────────────────────
    let dense_bytes = n_communities.saturating_mul(n_groups).saturating_mul(8);
    let use_dense = dense_bytes <= DENSE_MEMORY_BUDGET_BYTES;
    println!(
        "Swap refinement: {} nodes, {} communities, {} groups → {} backend ({:.1} MB)",
        total_nodes, n_communities, n_groups,
        if use_dense { "dense" } else { "sparse" },
        dense_bytes as f64 / 1e6,
    );

    // ── Sparse budget: B[g] = HashMap<h, target> ─────────────────────────────
    let mut budget_map: Vec<HashMap<usize, f64>> = (0..n_groups).map(|_| HashMap::new()).collect();
    for (&(sg, dg), &val) in &budget {
        if val <= 0 { continue; }
        let sg = sg as usize;
        let dg = dg as usize;
        if sg < n_groups && dg < n_groups {
            budget_map[sg].insert(dg, val as f64);
        }
    }

    // ── Composition storage ─────────────────────────────────────────────────
    let mut comp = if use_dense {
        Comp::Dense((0..n_communities).map(|_| vec![0i64; n_groups]).collect())
    } else {
        Comp::Sparse((0..n_communities).map(|_| HashMap::new()).collect())
    };

    let mut current_assign: Vec<usize> = Vec::with_capacity(total_nodes);
    for i in 0..total_nodes {
        let c = assignments_in[i] as usize;
        let g = node_groups_arr[i] as usize;
        if c < n_communities && g < n_groups {
            comp.add(c, g, 1);
        }
        current_assign.push(c);
    }

    // ── Inverted index: comm_groups[c] = sorted populated groups in c ────────
    let mut comm_groups: Vec<Vec<usize>> = (0..n_communities)
        .map(|c| {
            let mut v: Vec<usize> = (0..n_groups).filter(|&g| comp.get(c, g) > 0).collect();
            v.sort_unstable();
            v
        })
        .collect();

    // ── Achieved: A[g] = HashMap<h, count> (sparse, symmetric in pattern) ────
    let mut achieved: Vec<HashMap<usize, f64>> = (0..n_groups).map(|_| HashMap::new()).collect();
    for c in 0..n_communities {
        let groups_in_c = &comm_groups[c];
        for &g in groups_in_c {
            let cg = comp.get(c, g) as f64;
            for &h in groups_in_c {
                let ch = comp.get(c, h) as f64;
                *achieved[g].entry(h).or_insert(0.0) += cg * ch;
            }
        }
    }

    // ── Initial loss ─────────────────────────────────────────────────────────
    let initial_loss = compute_total_loss(&achieved, &budget_map, n_groups, overshoot_penalty);
    println!("Initial loss = {:.2}", initial_loss);

    // ── Node lookup: comm_group_nodes[(c, g)] = Vec<node_idx> ────────────────
    let mut comm_group_nodes: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for i in 0..total_nodes {
        let c = current_assign[i];
        let g = node_groups_arr[i] as usize;
        comm_group_nodes.entry((c, g)).or_insert_with(Vec::new).push(i);
    }

    if n_communities < 2 {
        println!("Only {} community — nothing to refine.", n_communities);
        let result: Vec<i64> = current_assign.iter().map(|&c| c as i64).collect();
        return Ok(PyArray1::from_owned_array_bound(py, Array1::from(result)));
    }

    // ── Pre-allocated buffers ────────────────────────────────────────────────
    let mut touched: Vec<usize> = Vec::with_capacity(256);

    // ── Bias index: list of communities containing each group ────────────────
    // comms_with_group[g] = Vec<c> for picking biased c2.
    let mut comms_with_group: Vec<Vec<usize>> = (0..n_groups).map(|_| Vec::new()).collect();
    for c in 0..n_communities {
        for &g in &comm_groups[c] {
            comms_with_group[g].push(c);
        }
    }

    let mut accepted: usize = 0;
    let mut current_loss = initial_loss;
    let report_every = (n_iterations / 10).max(1);

    // ── Main swap loop ───────────────────────────────────────────────────────
    for iter in 0..n_iterations {
        let progress = iter as f64 / n_iterations.max(1) as f64;
        let temperature = (temperature_start * (1.0 - progress)).max(0.0);

        // ── Pick c1, c2, g1, g2 (biased or random) ───────────────────────────
        let (c1, c2, g1, g2) = if rng.gen::<f64>() < biased_fraction {
            // Biased: c1 has g1 over-budget; c2 has g2 over-budget;
            // swapping puts g1 where it's needed and g2 where it's needed.
            match propose_biased(
                &comp, &achieved, &budget_map,
                &comm_groups, &comms_with_group,
                n_communities, &mut rng,
            ) {
                Some(t) => t,
                None => {
                    // Fall back to random if biased failed
                    match propose_random(&comm_groups, n_communities, &mut rng) {
                        Some(t) => t,
                        None => continue,
                    }
                }
            }
        } else {
            match propose_random(&comm_groups, n_communities, &mut rng) {
                Some(t) => t,
                None => continue,
            }
        };

        // ── Build `touched` = comm_groups[c1] ∪ comm_groups[c2] ──────────────
        touched.clear();
        merge_sorted_unique(&comm_groups[c1], &comm_groups[c2], &mut touched);

        // ── Pre-swap counts (4 lookups, then everything from `touched` loop) ─
        let c1_g1 = comp.get(c1, g1) as f64;
        let c1_g2 = comp.get(c1, g2) as f64;
        let c2_g1 = comp.get(c2, g1) as f64;
        let c2_g2 = comp.get(c2, g2) as f64;
        let c1_g1_new = c1_g1 - 1.0;
        let c1_g2_new = c1_g2 + 1.0;
        let c2_g1_new = c2_g1 + 1.0;
        let c2_g2_new = c2_g2 - 1.0;

        // ── Compute delta in a single pass over `touched` ────────────────────
        let mut delta_loss = 0.0f64;
        for &h in &touched {
            let (c1_h_old, c2_h_old) = if h == g1 {
                (c1_g1, c2_g1)
            } else if h == g2 {
                (c1_g2, c2_g2)
            } else {
                (comp.get(c1, h) as f64, comp.get(c2, h) as f64)
            };
            let c1_h_new = if h == g1 { c1_g1_new } else if h == g2 { c1_g2_new } else { c1_h_old };
            let c2_h_new = if h == g1 { c2_g1_new } else if h == g2 { c2_g2_new } else { c2_h_old };

            // A[g1, h]
            let delta_a1 = c1_g1_new * c1_h_new + c2_g1_new * c2_h_new
                         - c1_g1     * c1_h_old - c2_g1     * c2_h_old;
            if delta_a1 != 0.0 {
                let av = *achieved[g1].get(&h).unwrap_or(&0.0);
                let bv = *budget_map[g1].get(&h).unwrap_or(&0.0);
                delta_loss += cost_at(av + delta_a1, bv, overshoot_penalty)
                            - cost_at(av, bv, overshoot_penalty);
            }

            // A[g2, h]
            let delta_a2 = c1_g2_new * c1_h_new + c2_g2_new * c2_h_new
                         - c1_g2     * c1_h_old - c2_g2     * c2_h_old;
            if delta_a2 != 0.0 {
                let av = *achieved[g2].get(&h).unwrap_or(&0.0);
                let bv = *budget_map[g2].get(&h).unwrap_or(&0.0);
                delta_loss += cost_at(av + delta_a2, bv, overshoot_penalty)
                            - cost_at(av, bv, overshoot_penalty);
            }

            // A[h, g1] and A[h, g2] when h ∉ {g1, g2}
            if h != g1 && h != g2 {
                let delta_a3 = c1_h_old * c1_g1_new + c2_h_old * c2_g1_new
                             - c1_h_old * c1_g1     - c2_h_old * c2_g1;
                if delta_a3 != 0.0 {
                    let av = *achieved[h].get(&g1).unwrap_or(&0.0);
                    let bv = *budget_map[h].get(&g1).unwrap_or(&0.0);
                    delta_loss += cost_at(av + delta_a3, bv, overshoot_penalty)
                                - cost_at(av, bv, overshoot_penalty);
                }

                let delta_a4 = c1_h_old * c1_g2_new + c2_h_old * c2_g2_new
                             - c1_h_old * c1_g2     - c2_h_old * c2_g2;
                if delta_a4 != 0.0 {
                    let av = *achieved[h].get(&g2).unwrap_or(&0.0);
                    let bv = *budget_map[h].get(&g2).unwrap_or(&0.0);
                    delta_loss += cost_at(av + delta_a4, bv, overshoot_penalty)
                                - cost_at(av, bv, overshoot_penalty);
                }
            }
        }

        // ── Accept? ──────────────────────────────────────────────────────────
        let accept = if delta_loss < 0.0 {
            true
        } else if temperature > 1e-9 {
            (-delta_loss / temperature).exp() > rng.gen::<f64>()
        } else {
            false
        };
        if !accept { continue; }

        // ── Apply node moves ─────────────────────────────────────────────────
        let nodes_c1_g1 = comm_group_nodes.get_mut(&(c1, g1)).unwrap();
        let node_a = nodes_c1_g1.pop().unwrap();
        if nodes_c1_g1.is_empty() { comm_group_nodes.remove(&(c1, g1)); }

        let nodes_c2_g2 = comm_group_nodes.get_mut(&(c2, g2)).unwrap();
        let node_b = nodes_c2_g2.pop().unwrap();
        if nodes_c2_g2.is_empty() { comm_group_nodes.remove(&(c2, g2)); }

        comm_group_nodes.entry((c2, g1)).or_insert_with(Vec::new).push(node_a);
        comm_group_nodes.entry((c1, g2)).or_insert_with(Vec::new).push(node_b);
        current_assign[node_a] = c2;
        current_assign[node_b] = c1;

        // ── Update comp ──────────────────────────────────────────────────────
        comp.add(c1, g1, -1);
        comp.add(c1, g2,  1);
        comp.add(c2, g1,  1);
        comp.add(c2, g2, -1);

        // ── Update inverted indices ──────────────────────────────────────────
        // c1: g1 may have left (if c1_g1_new == 0), g2 may have arrived (if c1_g2 == 0)
        if c1_g1_new == 0.0 {
            comm_groups[c1].retain(|&x| x != g1);
            comms_with_group[g1].retain(|&x| x != c1);
        }
        if c1_g2 == 0.0 {
            let pos = comm_groups[c1].partition_point(|&x| x < g2);
            comm_groups[c1].insert(pos, g2);
            comms_with_group[g2].push(c1);
        }
        // c2: g2 may have left, g1 may have arrived
        if c2_g2_new == 0.0 {
            comm_groups[c2].retain(|&x| x != g2);
            comms_with_group[g2].retain(|&x| x != c2);
        }
        if c2_g1 == 0.0 {
            let pos = comm_groups[c2].partition_point(|&x| x < g1);
            comm_groups[c2].insert(pos, g1);
            comms_with_group[g1].push(c2);
        }

        // ── Apply achieved-matrix updates ────────────────────────────────────
        // Same pass structure as delta computation.
        for &h in &touched {
            let (c1_h, c2_h) = if h == g1 {
                (c1_g1, c2_g1)
            } else if h == g2 {
                (c1_g2, c2_g2)
            } else {
                (comp.get(c1, h) as f64, comp.get(c2, h) as f64)
            };
            let c1_h_new = if h == g1 { c1_g1_new } else if h == g2 { c1_g2_new } else { c1_h };
            let c2_h_new = if h == g1 { c2_g1_new } else if h == g2 { c2_g2_new } else { c2_h };

            // A[g1, h]
            let delta1 = c1_g1_new * c1_h_new + c2_g1_new * c2_h_new
                       - c1_g1     * c1_h     - c2_g1     * c2_h;
            if delta1 != 0.0 {
                let entry = achieved[g1].entry(h).or_insert(0.0);
                *entry += delta1;
                if *entry == 0.0 { achieved[g1].remove(&h); }
            }

            // A[g2, h]
            let delta2 = c1_g2_new * c1_h_new + c2_g2_new * c2_h_new
                       - c1_g2     * c1_h     - c2_g2     * c2_h;
            if delta2 != 0.0 {
                let entry = achieved[g2].entry(h).or_insert(0.0);
                *entry += delta2;
                if *entry == 0.0 { achieved[g2].remove(&h); }
            }

            if h != g1 && h != g2 {
                let delta3 = c1_h * c1_g1_new + c2_h * c2_g1_new
                           - c1_h * c1_g1     - c2_h * c2_g1;
                if delta3 != 0.0 {
                    let entry = achieved[h].entry(g1).or_insert(0.0);
                    *entry += delta3;
                    if *entry == 0.0 { achieved[h].remove(&g1); }
                }

                let delta4 = c1_h * c1_g2_new + c2_h * c2_g2_new
                           - c1_h * c1_g2     - c2_h * c2_g2;
                if delta4 != 0.0 {
                    let entry = achieved[h].entry(g2).or_insert(0.0);
                    *entry += delta4;
                    if *entry == 0.0 { achieved[h].remove(&g2); }
                }
            }
        }

        current_loss += delta_loss;
        accepted += 1;

        if (iter + 1) % report_every == 0 {
            println!(
                "  iter {}/{}: loss={:.2}, accept={:.1}%, T={:.3}",
                iter + 1, n_iterations, current_loss,
                100.0 * accepted as f64 / (iter + 1) as f64, temperature
            );
        }
    }

    println!(
        "Refinement done: {}/{} accepted ({:.1}%), final loss={:.2}, reduction={:.1}%",
        accepted, n_iterations,
        100.0 * accepted as f64 / n_iterations.max(1) as f64,
        current_loss,
        if initial_loss > 0.0 { 100.0 * (initial_loss - current_loss) / initial_loss } else { 0.0 }
    );

    let result: Vec<i64> = current_assign.iter().map(|&c| c as i64).collect();
    Ok(PyArray1::from_owned_array_bound(py, Array1::from(result)))
}


/// Pure random proposal — pick two communities, one populated group from each.
#[inline]
fn propose_random(
    comm_groups: &[Vec<usize>],
    n_communities: usize,
    rng: &mut StdRng,
) -> Option<(usize, usize, usize, usize)> {
    if n_communities < 2 { return None; }
    let c1 = rng.gen_range(0..n_communities);
    let c2 = rng.gen_range(0..n_communities);
    if c1 == c2 { return None; }
    if comm_groups[c1].is_empty() || comm_groups[c2].is_empty() { return None; }
    let g1 = comm_groups[c1][rng.gen_range(0..comm_groups[c1].len())];
    let g2 = comm_groups[c2][rng.gen_range(0..comm_groups[c2].len())];
    if g1 == g2 { return None; }
    Some((c1, c2, g1, g2))
}


/// Biased proposal: pick c1 with a group g1 that's *over*-contributing relative
/// to budget, and c2 with a group g2 that's also over-contributing. Swapping
/// them is likely to reduce loss.
///
/// Heuristic: sample (c1, g1) by sampling g1 first (uniform from populated
/// groups in any community), then sampling c1 from the communities containing
/// g1 weighted by how much g1's presence in c1 contributes to over-budget.
/// Same for (c2, g2). Cheap approximation, not a global optimiser.
fn propose_biased(
    comp: &Comp,
    achieved: &[HashMap<usize, f64>],
    budget_map: &[HashMap<usize, f64>],
    comm_groups: &[Vec<usize>],
    comms_with_group: &[Vec<usize>],
    n_communities: usize,
    rng: &mut StdRng,
) -> Option<(usize, usize, usize, usize)> {
    if n_communities < 2 { return None; }

    // Sample two random communities first; pick a group from each that
    // is "most over-budget" within that community. This is much cheaper than
    // a global most-over-budget search and still gives directional guidance.
    let c1 = rng.gen_range(0..n_communities);
    let c2 = rng.gen_range(0..n_communities);
    if c1 == c2 || comm_groups[c1].is_empty() || comm_groups[c2].is_empty() {
        return None;
    }

    // For c1, find a g where achieved[g, *] is over budget (sample weighted).
    let g1 = sample_overbudget_group(c1, comp, achieved, budget_map, comm_groups, rng)?;
    let g2 = sample_overbudget_group(c2, comp, achieved, budget_map, comm_groups, rng)?;
    if g1 == g2 { return None; }

    // Sanity: both groups must actually be in their communities for a valid swap
    if comp.get(c1, g1) == 0 || comp.get(c2, g2) == 0 { return None; }

    let _ = comms_with_group; // reserved for stronger biasing later
    Some((c1, c2, g1, g2))
}

/// Sample a group from community c, weighted by how much that group's pairs
/// in c are currently over-budget. Returns None if no over-budget group exists.
fn sample_overbudget_group(
    c: usize,
    comp: &Comp,
    achieved: &[HashMap<usize, f64>],
    budget_map: &[HashMap<usize, f64>],
    comm_groups: &[Vec<usize>],
    rng: &mut StdRng,
) -> Option<usize> {
    let groups = &comm_groups[c];
    if groups.is_empty() { return None; }

    // Cheap weight: for each g in c, weight = overshoot of A[g, *] summed over
    // h in c (since these are the pairs g currently contributes to).
    let mut weights: Vec<f64> = Vec::with_capacity(groups.len());
    let mut total = 0.0f64;
    for &g in groups {
        let cg = comp.get(c, g) as f64;
        if cg == 0.0 { weights.push(0.0); continue; }
        let mut w = 0.0;
        for &h in groups {
            let av = *achieved[g].get(&h).unwrap_or(&0.0);
            let bv = *budget_map[g].get(&h).unwrap_or(&0.0);
            if av > bv { w += av - bv; }
        }
        weights.push(w);
        total += w;
    }

    if total <= 0.0 {
        // No over-budget pair in this community; fall back to uniform
        return Some(groups[rng.gen_range(0..groups.len())]);
    }

    // Weighted sample
    let target = rng.gen::<f64>() * total;
    let mut cum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= target { return Some(groups[i]); }
    }
    Some(*groups.last().unwrap())
}


/// Total loss over all (g, h) pairs where achieved or budget is non-zero.
fn compute_total_loss(
    achieved: &[HashMap<usize, f64>],
    budget_map: &[HashMap<usize, f64>],
    n_groups: usize,
    overshoot_penalty: f64,
) -> f64 {
    let mut total = 0.0;
    for g in 0..n_groups {
        for (&h, &av) in &achieved[g] {
            let bv = budget_map[g].get(&h).copied().unwrap_or(0.0);
            total += cost_at(av, bv, overshoot_penalty);
        }
        for (&h, &bv) in &budget_map[g] {
            if !achieved[g].contains_key(&h) {
                total += cost_at(0.0, bv, overshoot_penalty);
            }
        }
    }
    total
}


#[inline(always)]
fn cost_at(achieved_val: f64, budget_val: f64, overshoot_penalty: f64) -> f64 {
    let diff = achieved_val - budget_val;
    if diff > 0.0 { overshoot_penalty * diff * diff } else { diff * diff }
}


/// Merge two sorted Vecs into `out`, deduplicating the union. O(|a| + |b|).
#[inline]
fn merge_sorted_unique(a: &[usize], b: &[usize], out: &mut Vec<usize>) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less    => { out.push(a[i]); i += 1; }
            std::cmp::Ordering::Greater => { out.push(b[j]); j += 1; }
            std::cmp::Ordering::Equal   => { out.push(a[i]); i += 1; j += 1; }
        }
    }
    while i < a.len() { out.push(a[i]); i += 1; }
    while j < b.len() { out.push(b[j]); j += 1; }
}

/// Python module
#[pymodule]
fn asnu_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_nodes, m)?)?;
    m.add_function(wrap_pyfunction!(run_edge_creation, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(process_nodes_capacity_fast, m)?)?;
    m.add_function(wrap_pyfunction!(refine_communities_swap, m)?)?;
    Ok(())
}