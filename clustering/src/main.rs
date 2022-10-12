use itertools::Itertools;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::{prelude::*, BufReader, BufWriter};
use std::path::Path;

mod clustering;
use crate::clustering::*;

/*
fn cluster_until_n_clusters(n: usize, points: &Vec<(f32, f32)>) -> Vec<Cluster> {
    let mut clusters: Vec<Cluster> = points.iter().map(|p| Cluster::new(p)).collect();
    {
        let mut matrix = DistanceMatrix::with_clusters(
            &mut clusters,
            DistanceMethod::Euclidean,
            ClusteringMethod::SingleLinkage,
        );
        matrix.get_distances();
        while matrix.clusters.as_ref().unwrap().len() > n {
            matrix.cluster_closest();
        }
    }
    // println!("{:?}", clusters);
    return clusters;
}*/

fn main() -> Result<(), std::io::Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "A file path argument must be supplied",
        ));
    }
    let file_path = Path::new(&args[1]);
    let file = File::open(&file_path)?;
    let reader = BufReader::new(file);

    let points: Vec<(f32, f32)> = reader
        .lines()
        .map(|x| {
            x.unwrap()
                .split(';')
                .map(|y| y.parse::<f32>().unwrap())
                .collect_tuple()
                .unwrap()
        })
        .collect();

    {
        let max_clusters = [5usize, 3usize, 2usize];
        // cluster_until_n_clusters(3usize, &points);
        let distance_methods = [DistanceMethod::Euclidean, DistanceMethod::Manhattan];
        let clustering_methods = [
            ClusteringMethod::SingleLinkage,
            ClusteringMethod::CompleteLinkage,
        ];
        for distance_method in distance_methods {
            for clustering_method in clustering_methods {
                let dist_str = match distance_method {
                    DistanceMethod::Euclidean => "euclid",
                    DistanceMethod::Manhattan => "manhat",
                };
                let clust_str = match clustering_method {
                    ClusteringMethod::SingleLinkage => "singl",
                    ClusteringMethod::CompleteLinkage => "comp",
                };
                let clusters: Vec<Cluster> = points.iter().map(Cluster::new).collect();
                let mut matrix =
                    DistanceMatrix::with_clusters(clusters, distance_method, clustering_method);
                matrix.get_distances();
                for n in max_clusters {
                    while matrix.clusters.as_ref().unwrap().len() > n {
                        matrix.cluster_closest();
                    }
                    let basename = file_path.file_stem().unwrap().to_str().unwrap();
                    let file = OpenOptions::new().write(true).create(true).open(format!(
                        "./{base}-{clu}-{dist}-clusters-{n}.csv",
                        base = basename,
                        clu = clust_str,
                        dist = dist_str,
                        n = n
                    ))?;
                    let mut writer = BufWriter::new(file);
                    let mut cid = 1;
                    writer.write("x;y;cluster\n".as_bytes())?;
                    for cluster in matrix.clusters.as_ref().unwrap().iter() {
                        for p in cluster.points.iter() {
                            let (x, y) = p;
                            writer.write_fmt(format_args!(
                                "{x};{y};C{cluster_id}\n",
                                x = *x,
                                y = *y,
                                cluster_id = cid,
                            ))?;
                        }
                        cid += 1;
                    }
                }
            }
        }
    }

    Ok(())
}
