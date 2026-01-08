from bs4 import BeautifulSoup
from test_3D_clusterworker import ClusterEstimation

# Load and Parse KML
kml_file = 'test_3d_estimation.kml'
try:
    with open(kml_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'xml')
except FileNotFoundError:
    print(f"Error: {kml_file} not found.")
    exit()

extracted_points = []
placemarks = soup.find_all('Placemark')

for p in placemarks:
    coords = p.find('coordinates')
    if coords:
        # KML coords are often: lon,lat,alt (space separated if multiple)
        raw_text = coords.text.strip()
        for entry in raw_text.split():
            parts = entry.split(',')
            if len(parts) >= 2:
                try:
                    lon, lat, alt = float(parts[0]), float(parts[1]), float(parts[2])
                    extracted_points.append([lat, lon, alt])
                except ValueError:
                    continue

print(f"Extracted {len(extracted_points)} points.")
# print(extracted_points)
# Initialize and Run Clustering
success, clusterInstance = ClusterEstimation.create(
    min_activation_threshold=5, # Requires at least 5 points to start
    min_new_points_to_run=1, 
    max_num_components=10, 
    random_state=42, 
    min_points_per_cluster=2,
)

if success and clusterInstance: 
    did_run, clusters = clusterInstance.run(extracted_points)
    if did_run:
        print(f"Successfully identified {len(clusters)} clusters:")
        for i, (mean, weight, cov) in enumerate(clusters):
            print(f"Cluster {i+1}: Center={mean}, Weight={weight:.2f}, Variance={cov:.6f}")
    else:
        print("Clustering did not run (insufficient points or not converged).")
else: 
    print("Failed to initialize ClusterEstimation (check your thresholds).")