import os
import pandas as pd
import trimesh

def is_already_processed(object_name, food_type, processed_set):
    return (object_name, food_type) in processed_set

def process_mesh(path, processed_set):
    if not path.endswith('.obj'):
        return None
    try:
        food_type = os.path.basename(os.path.dirname(os.path.dirname(path)))
        object_name = os.path.basename(os.path.dirname(path))

        if is_already_processed(object_name, food_type, processed_set):
            return None
        
        mesh = trimesh.load(path, force='mesh', process=False, fast_load=True)

        if mesh.is_empty or mesh.volume == 0:
            raise ValueError("Empty mesh or zero volume.")

        volume_ml = mesh.volume * 1e-3  # Assuming mesh.volume is in mm^3
        result = {
            'Object_name': object_name,
            'Food_Type': food_type,
            'Volume(ml)': round(volume_ml, 2),
            'Path': path
        }
        return result
    except Exception as e:
        with open('error_log.txt', 'a') as f:
            f.write(f"{path}: {e}\n")
        return None

def collect_processed_set(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return set(zip(df['Object_name'], df['Food_Type']))
    return set()

def main(root_dir, output_csv='volumes.csv'):
    processed_set = collect_processed_set(output_csv)
    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.obj'):
                obj_path = os.path.join(dirpath, fname)
                result = process_mesh(obj_path, processed_set)
                if result:
                    results.append(result)

    if results:
        df = pd.DataFrame(results)
        header = not os.path.exists(output_csv)
        df.to_csv(output_csv, mode='a', header=header, index=False)
        print(f"Wrote {len(results)} new entries to {output_csv}")
    else:
        print("No new volumes calculated.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python volume_script.py /path/to/root_dir [output_csv]")
    else:
        root = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else 'volumes.csv'
        main(root, output)
