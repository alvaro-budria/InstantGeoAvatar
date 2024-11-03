import os
import os.path as osp
import shutil
import numpy as np
import pickle as pkl
from PIL import Image


def convert_mesh(mesh_file, mesh_name, save_folder):
    mesh_data = pkl.load(open(mesh_file, 'rb'), encoding='latin1')
    # mesh_name = osp.basename(mesh_file)
    obj_name = mesh_name + '.obj'
    mtl_name = mesh_name +  '.mtl'
    texture_name = mesh_name + '.jpg'
    # write obj
    with open(osp.join(save_folder, obj_name), 'w') as f:
        f.write("#OBJ\n")
        f.write(f"#{len(mesh_data['vertices'])} pos\n")
        for v in mesh_data['vertices']:
            f.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
        f.write(f"#{len(mesh_data['normals'])} norm\n")
        for vn in mesh_data['normals']:
            f.write("vn %.4f %.4f %.4f\n" % (vn[0], vn[1], vn[2]))
        f.write(f"#{len(mesh_data['uvs'])} tex\n")
        for vt in mesh_data['uvs']:
            f.write("vt %.4f %.4f\n" % (vt[0], vt[1]))
        f.write(f"#{len(mesh_data['faces'])} faces\n")
        f.write("mtllib {}\n".format(mtl_name))
        f.write("usemtl atlasTextureMap\n")
        for fc in mesh_data['faces']:
            f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (fc[0]+1, fc[0]+1, fc[0]+1, fc[1]+1, fc[1]+1, fc[1]+1, fc[2]+1, fc[2]+1, fc[2]+1))

    # write mtl
    with open(osp.join(save_folder, mtl_name), 'w') as f:
        f.write("newmtl atlasTextureMap\n")
        s = 'map_Kd {}\n'.format(mtl_name)  # map to image
        f.write(s)

    # write texture
    tex = pkl.load(open(mesh_file.replace('mesh-', 'atlas-'), 'rb'), encoding='latin1')
    uv_map = Image.fromarray(tex).transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
    uv_map.save(osp.join(save_folder, texture_name))


def main(sequences):

    for sequence, take_numbers in sequences.items():
        print(f"Processing sequence {sequence}")
        source_directory = sequence

        with open(os.path.join(source_directory, "gender.txt"), 'r') as file:
            gender = file.read().strip()
        # Print the gender read from the file
        print("Gender:", gender)

        # Define the target directory where images will be stored
        target_directory = f"XHumans/{sequence}"

        take_numbers_all = take_numbers['train'] + take_numbers['test']

        # Create the target directory if it doesn't exist
        os.makedirs(target_directory, exist_ok=True)

        # Define subdirectories for images and depth images
        image_subdirectory = os.path.join(target_directory, "images")
        depth_subdirectory = os.path.join(target_directory, "masks")
        mesh_subdirectory = os.path.join(target_directory, "meshes_obj")
        os.makedirs(image_subdirectory, exist_ok=True)
        os.makedirs(depth_subdirectory, exist_ok=True)
        os.makedirs(mesh_subdirectory, exist_ok=True)

        # Initialize an index for reindexing the images
        index = 0
        # Dictionary to store SMPL parameters
        parameters_dict = {
            'global_orient': [], 'transl': [], 'body_pose': [],
            'intrinsic': [], 'extrinsic': [],
            'height': None, 'width': None, 'gender': gender,
        }
        # List to store index limits of each subsequence
        index_limits = []

        # Iterate over take numbers
        for take_number in take_numbers_all:
            sequence_type = 'train' if take_number in take_numbers['train'] else 'test'
            # Check in 'train' directory first
            image_directory = os.path.join(source_directory, "train", f"Take{take_number}", "render", "image")
            depth_directory = os.path.join(source_directory, "train", f"Take{take_number}", "render", "depth")
            meshes_directory = os.path.join(source_directory, "train", f"Take{take_number}", "meshes_pkl")
            smpl_directory = os.path.join(source_directory, "train", f"Take{take_number}", "SMPL")
            camera_path = os.path.join(source_directory, "train", f"Take{take_number}", "render", "cameras.npz")
            print('image_directory', image_directory)
            if not os.path.exists(image_directory):
                # If not found in 'train', check in 'test' directory
                image_directory = os.path.join(source_directory, "test", f"Take{take_number}", "render", "image")
                depth_directory = os.path.join(source_directory, "test", f"Take{take_number}", "render", "depth")
                meshes_directory = os.path.join(source_directory, "test", f"Take{take_number}", "meshes_pkl")
                smpl_directory = os.path.join(source_directory, "test", f"Take{take_number}", "SMPL")
                camera_path = os.path.join(source_directory, "test", f"Take{take_number}", "render", "cameras.npz")

            # Check if the image directory exists
            if os.path.exists(image_directory):
                index_original = index
                start_index = index

                # Iterate over images in the 'image' directory
                for image_file in sorted(os.listdir(image_directory)):
                    # Construct the source and target paths for the image
                    source_image_path = os.path.join(image_directory, image_file)
                    target_image_path = os.path.join(image_subdirectory, f"{index:05d}.png")

                    # Copy the image to the target directory and rename it
                    shutil.copyfile(source_image_path, target_image_path)
                    index += 1

                end_index = index - 1
                index_limits.append({'take_number': take_number, 'start': start_index, 'end': end_index, 'type': sequence_type})

                index = index_original
                for depth_file in sorted(os.listdir(depth_directory)):
                    # Load depth image
                    depth_image_path = os.path.join(depth_directory, depth_file)
                    with Image.open(depth_image_path) as depth_image:
                        # Convert to numpy array and normalize to range [0, 1]
                        depth_array = np.array(depth_image) / 255.0
                        # Binaraize the depth image
                        depth_array[depth_array >= 0.5] = 1
                        # Flip depth values (0 to 1 and vice versa)
                        flipped_depth_array = np.abs(depth_array - 1)
                        # Convert back to PIL image
                        flipped_depth_image = Image.fromarray((flipped_depth_array * 255).astype(np.uint8))
                        # Save the depth image, with 5 digits index
                        flipped_depth_path = os.path.join(depth_subdirectory, f"{index:05d}.png")
                        flipped_depth_image.save(flipped_depth_path)
                        index += 1
                parameters_dict['height'], parameters_dict['width'] = depth_array.shape

                index = index_original
                for mesh_file in sorted(os.listdir(meshes_directory)):
                    if mesh_file.count('mesh') == 1:
                        mesh_file_path = os.path.join(meshes_directory, mesh_file)
                        mesh_name = f"{index:05d}"
                        convert_mesh(mesh_file_path, mesh_name, mesh_subdirectory)
                        index += 1

                index = index_original
                for smpl_file in sorted(os.listdir(smpl_directory)):
                    if smpl_file.endswith('.pkl'):
                        # Load smpl file
                        smpl_file_path = os.path.join(smpl_directory, smpl_file)
                        with open(smpl_file_path, 'rb') as f:
                            smpl_data = pkl.load(f)
                        # Retrieve parameters from smpl data
                        parameters_dict['global_orient'].append(smpl_data['global_orient'])
                        parameters_dict['transl'].append(smpl_data['transl'])
                        parameters_dict['body_pose'].append(smpl_data['body_pose'])
                        index += 1

                # Load camera parameters
                camera_data = np.load(camera_path)
                intrinsic = camera_data['intrinsic']
                extrinsic = camera_data['extrinsic']
                intrinsic = np.expand_dims(intrinsic, axis=0).repeat(len(extrinsic), axis=0)
                # Append camera parameters to the container
                parameters_dict['intrinsic'].append(intrinsic)
                parameters_dict['extrinsic'].append(extrinsic)

        for key in parameters_dict:
            if key in ['intrinsic', 'extrinsic']:
                parameters_dict[key] = np.concatenate(parameters_dict[key], axis=0)
            elif key in ['gender', 'height', 'width']:
                parameters_dict[key] = parameters_dict[key]
            else:
                parameters_dict[key] = np.stack(parameters_dict[key])
        betas = np.load(os.path.join(source_directory, "mean_shape_smpl.npy"))
        parameters_dict['betas'] = betas
        # Save the parameters dictionary
        parameters_path = os.path.join(target_directory, "smpl_camera_params.npz")
        np.savez(parameters_path, **parameters_dict)

        # Write index limits to a CSV file
        csv_file_path = os.path.join(target_directory, f"{sequence}_index_limits.csv")
        with open(csv_file_path, 'w') as f:
            # Write all the dictionary keys in a file with commas separated.
            f.write(','.join(index_limits[0].keys()))
            f.write('\n') # Add a new line
            for row in index_limits:
                # Write the values in a row.
                f.write(','.join(str(x) for x in row.values()))
                f.write('\n') # Add a new line

        print("Images have been copied, normalized, and reindexed successfully. SMPL parameters have been saved.")


if __name__ == "__main__":
    sequences = {
        '00016': {'train': [1, 13], 'test': [2, 5]},
        '00017': {'train': [13], 'test': [4, 5]},
        '00018': {'train': [6], 'test': [7]},
        '00019': {'train': [12], 'test': [6, 7]},
        '00020': {'train': [9], 'test': [15, 16]},
        '00021': {'train': [3], 'test': [5, 6]},
        '00024': {'train': [10], 'test': [5, 9]},
        '00025': {'train': [8], 'test': [2, 4]},
        '00027': {'train': [3], 'test': [4, 12]},
        '00028': {'train': [7, 8], 'test': [11, 15]},
        '00034': {'train': [12], 'test': [5, 13]},
        '00035': {'train': [6], 'test': [3,7]},
        '00036': {'train': [8], 'test': [6, 14]},
        '00039': {'train': [6], 'test': [9, 12]},
        '00041': {'train': [6], 'test': [1, 12]},
        '00058': {'train': [21], 'test': [24, 27]},
        '00085': {'train': [2], 'test': [7, 8]},
        '00086': {'train': [9], 'test': [3, 13]},
        '00087': {'train': [4], 'test': [7, 13]},
        '00088': {'train': [8], 'test': [14, 6]},
    }
    main(sequences)
