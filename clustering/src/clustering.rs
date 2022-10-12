#[derive(Debug)]
pub struct Cluster<'a> {
    pub points: Vec<&'a (f32, f32)>,
}

impl<'a> Cluster<'a> {
    pub fn new(p: &'a (f32, f32)) -> Self {
        Cluster { points: vec![p] }
    }

    pub fn merge(&mut self, other: &Cluster<'a>) {
        self.points.extend(other.points.iter());
    }
}

#[derive(Clone, Copy)]
pub enum ClusteringMethod {
    SingleLinkage,
    CompleteLinkage,
}

#[derive(Clone, Copy)]
pub enum DistanceMethod {
    Euclidean,
    Manhattan,
}

pub struct DistanceMatrix<'a> {
    matrix: Vec<Vec<f32>>,
    dist_method: DistanceMethod,
    cluster_method: ClusteringMethod,
    pub clusters: Option<Vec<Cluster<'a>>>,
}

impl<'a> DistanceMatrix<'a> {
    pub fn new(
        size: usize,
        distance_method: DistanceMethod,
        cluster_method: ClusteringMethod,
    ) -> Self {
        let matrix = vec![vec![f32::INFINITY; size]; size];

        DistanceMatrix {
            cluster_method,
            dist_method: distance_method,
            matrix,
            clusters: None,
        }
    }

    pub fn with_clusters(
        clusters: Vec<Cluster<'a>>,
        distance_method: DistanceMethod,
        cluster_method: ClusteringMethod,
    ) -> Self {
        let size = clusters.len();
        let matrix = vec![vec![f32::INFINITY; size]; size];

        DistanceMatrix {
            cluster_method,
            dist_method: distance_method,
            matrix,
            clusters: Some(clusters),
        }
    }

    fn get_point_distance(&self, p1: &(f32, f32), p2: &(f32, f32)) -> f32 {
        match self.dist_method {
            DistanceMethod::Euclidean => {
                (p1.0 - p2.0) * (p1.0 - p2.0) + (p1.1 - p2.1) * (p1.1 - p2.1)
            }
            DistanceMethod::Manhattan => (p1.0 - p2.0).abs() + (p1.1 - p2.1).abs(),
        }
    }

    fn calc_distance(&self, c1: &Cluster, c2: &Cluster) -> f32 {
        let mut distances = vec![f32::INFINITY; c1.points.len() * c2.points.len()];
        let mut x = 0usize;
        for p1 in c1.points.iter() {
            for p2 in c2.points.iter() {
                distances[x] = self.get_point_distance(p1, p2);
                x += 1;
            }
        }

        match self.cluster_method {
            ClusteringMethod::SingleLinkage => {
                distances.iter().fold(f32::INFINITY, |a, &b| a.min(b))
            }
            ClusteringMethod::CompleteLinkage => distances.iter().fold(-1.0, |a, &b| a.max(b)),
        }
    }

    pub fn get_distances(&mut self) {
        if let Some(points) = &self.clusters {
            for (y, p1) in points.iter().enumerate() {
                for (x, p2) in points.iter().enumerate() {
                    if self.matrix[y][x] != f32::INFINITY || x == y {
                        continue;
                    }

                    let distance = self.calc_distance(p1, p2);
                    self.matrix[y][x] = distance;
                    self.matrix[x][y] = distance;
                }
            }
        } else {
            panic!("No points ref!");
        }
    }

    pub fn cluster_closest(&mut self) {
        let mut smallest = f32::INFINITY;
        let mut smallest_idx = 0usize;
        let mut cluster_a_idx = 0usize;
        for (y, row) in self.matrix.iter().enumerate() {
            let (idx, min) = row
                .iter()
                .enumerate()
                .filter(|&(x, _)| x != y)
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, &min)| (index, min))
                .unwrap();
            if min < smallest {
                smallest = min;
                smallest_idx = idx;
                cluster_a_idx = y;
            }
        }
        if let Some(clusters) = &mut self.clusters {
            if smallest_idx < cluster_a_idx {
                let (head, tail) = clusters.split_at_mut(smallest_idx + 1);
                let ca = &mut tail[cluster_a_idx - smallest_idx - 1];
                let cb = &mut head.get(smallest_idx);
                ca.merge(cb.unwrap());
            } else {
                let (head, tail) = clusters.split_at_mut(cluster_a_idx + 1);
                let ca = &mut head[cluster_a_idx];
                let cb = &mut tail.get(smallest_idx - cluster_a_idx - 1);
                ca.merge(cb.unwrap());
            }
            clusters.remove(smallest_idx);
        }

        // remove smallest row and col
        self.matrix.remove(smallest_idx);
        for row in self.matrix.iter_mut() {
            row.remove(smallest_idx);
        }

        println!("Clusters: {}", self.matrix.len());

        if let Some(clusters) = &self.clusters {
            for (x, p2) in clusters.iter().enumerate() {
                if x == cluster_a_idx {
                    continue;
                }

                let distance = self.calc_distance(&clusters[cluster_a_idx], p2);
                self.matrix[cluster_a_idx][x] = distance;
                self.matrix[x][cluster_a_idx] = distance;
            }
        }
        // self.get_distances();
    }
}
