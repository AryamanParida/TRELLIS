# import torch
# from ...modules.sparse import SparseTensor
# from easydict import EasyDict as edict
# from .utils_cube import *
# from .flexicubes.flexicubes import FlexiCubes


# class MeshExtractResult:
#     def __init__(self,
#         vertices,
#         faces,
#         vertex_attrs=None,
#         res=64
#     ):
#         self.vertices = vertices
#         self.faces = faces.long()
#         self.vertex_attrs = vertex_attrs
#         self.face_normal = self.comput_face_normals(vertices, faces)
#         self.res = res
#         self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

#         # training only
#         self.tsdf_v = None
#         self.tsdf_s = None
#         self.reg_loss = None
        
#     def comput_face_normals(self, verts, faces):
#         i0 = faces[..., 0].long()
#         i1 = faces[..., 1].long()
#         i2 = faces[..., 2].long()

#         v0 = verts[i0, :]
#         v1 = verts[i1, :]
#         v2 = verts[i2, :]
#         face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
#         face_normals = torch.nn.functional.normalize(face_normals, dim=1)
#         # print(face_normals.min(), face_normals.max(), face_normals.shape)
#         return face_normals[:, None, :].repeat(1, 3, 1)
                
#     def comput_v_normals(self, verts, faces):
#         i0 = faces[..., 0].long()
#         i1 = faces[..., 1].long()
#         i2 = faces[..., 2].long()

#         v0 = verts[i0, :]
#         v1 = verts[i1, :]
#         v2 = verts[i2, :]
#         face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
#         v_normals = torch.zeros_like(verts)
#         v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
#         v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
#         v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

#         v_normals = torch.nn.functional.normalize(v_normals, dim=1)
#         return v_normals   


# class SparseFeatures2Mesh:
#     def __init__(self, device="cuda", res=64, use_color=True):
#         '''
#         a model to generate a mesh from sparse features structures using flexicube
#         '''
#         super().__init__()
#         self.device=device
#         self.res = res
#         self.mesh_extractor = FlexiCubes(device=device)
#         self.sdf_bias = -1.0 / res
#         verts, cube = construct_dense_grid(self.res, self.device)
#         self.reg_c = cube.to(self.device)
#         self.reg_v = verts.to(self.device)
#         self.use_color = use_color
#         self._calc_layout()
    
#     def _calc_layout(self):
#         LAYOUTS = {
#             'sdf': {'shape': (8, 1), 'size': 8},
#             'deform': {'shape': (8, 3), 'size': 8 * 3},
#             'weights': {'shape': (21,), 'size': 21}
#         }
#         if self.use_color:
#             '''
#             6 channel color including normal map
#             '''
#             LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
#         self.layouts = edict(LAYOUTS)
#         start = 0
#         for k, v in self.layouts.items():
#             v['range'] = (start, start + v['size'])
#             start += v['size']
#         self.feats_channels = start
        
#     def get_layout(self, feats : torch.Tensor, name : str):
#         if name not in self.layouts:
#             return None
#         return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
#     def __call__(self, cubefeats : SparseTensor, training=False):
#         """
#         Generates a mesh based on the specified sparse voxel structures.
#         Args:
#             cube_attrs [Nx21] : Sparse Tensor attrs about cube weights
#             verts_attrs [Nx10] : [0:1] SDF [1:4] deform [4:7] color [7:10] normal 
#         Returns:
#             return the success tag and ni you loss, 
#         """
#         # add sdf bias to verts_attrs
#         coords = cubefeats.coords[:, 1:]
#         feats = cubefeats.feats
        
#         sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
#         sdf += self.sdf_bias
#         v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
#         v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
#         v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
#         weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
#         if self.use_color:
#             sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
#         else:
#             sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
#             colors_d = None
            
#         x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
#         vertices, faces, L_dev, colors = self.mesh_extractor(
#             voxelgrid_vertices=x_nx3,
#             scalar_field=sdf_d,
#             cube_idx=self.reg_c,
#             resolution=self.res,
#             beta=weights_d[:, :12],
#             alpha=weights_d[:, 12:20],
#             gamma_f=weights_d[:, 20],
#             voxelgrid_colors=colors_d,
#             training=training)
        
#         mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
#         if training:
#             if mesh.success:
#                 reg_loss += L_dev.mean() * 0.5
#             reg_loss += (weights[:,:20]).abs().mean() * 0.2
#             mesh.reg_loss = reg_loss
#             mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
#             mesh.tsdf_s = v_attrs[:, 0]
#         return mesh

import torch
from ...modules.sparse import SparseTensor
from easydict import EasyDict as edict
from .utils_cube import *
from .flexicubes.flexicubes import FlexiCubes


class MeshExtractResult:
    def __init__(self,
        vertices,
        faces,
        vertex_attrs=None,
        res=64
    ):
        self.vertices = vertices
        self.faces = faces.long()
        self.vertex_attrs = vertex_attrs
        self.face_normal = self.comput_face_normals(vertices, faces)
        self.res = res
        self.success = (vertices.shape[0] != 0 and faces.shape[0] != 0)

        # training only
        self.tsdf_v = None
        self.tsdf_s = None
        self.reg_loss = None
        
    def comput_face_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        return face_normals[:, None, :].repeat(1, 3, 1)
                
    def comput_v_normals(self, verts, faces):
        i0 = faces[..., 0].long()
        i1 = faces[..., 1].long()
        i2 = faces[..., 2].long()

        v0 = verts[i0, :]
        v1 = verts[i1, :]
        v2 = verts[i2, :]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        v_normals = torch.zeros_like(verts)
        v_normals.scatter_add_(0, i0[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i1[..., None].repeat(1, 3), face_normals)
        v_normals.scatter_add_(0, i2[..., None].repeat(1, 3), face_normals)

        v_normals = torch.nn.functional.normalize(v_normals, dim=1)
        return v_normals   


class SparseFeatures2Mesh:
    def __init__(self, device="cuda", res=64, use_color=True, enable_multi_gpu=None):
        '''
        A model to generate a mesh from sparse features structures using flexicube
        
        Args:
            device: Primary device for computation
            res: Resolution of the mesh grid
            use_color: Whether to use color features
            enable_multi_gpu: If True and multiple GPUs available, use model parallelism
        '''
        super().__init__()
        self.device = device
        self.res = res
        self.use_color = use_color
        
        # Auto-detect multi-GPU setup
        if enable_multi_gpu is None:
            enable_multi_gpu = torch.cuda.device_count() > 1
        
        self.enable_multi_gpu = enable_multi_gpu
        
        if self.enable_multi_gpu and torch.cuda.device_count() > 1:
            print(f"🚀 Enabling multi-GPU mesh extraction with {torch.cuda.device_count()} GPUs")
            self._init_multi_gpu()
        else:
            print(f"📱 Using single GPU mesh extraction on {device}")
            self._init_single_gpu()
        
        self._calc_layout()
    
    def _init_single_gpu(self):
        """Initialize for single GPU operation (original behavior)."""
        self.mesh_extractor = FlexiCubes(device=self.device)
        self.sdf_bias = -1.0 / self.res
        verts, cube = construct_dense_grid(self.res, self.device)
        self.reg_c = cube.to(self.device)
        self.reg_v = verts.to(self.device)
    
    def _init_multi_gpu(self):
        """Initialize for multi-GPU operation."""
        self.device_0 = torch.device('cuda:0')
        self.device_1 = torch.device('cuda:1')
        
        # Create mesh extractors on both devices
        self.mesh_extractor_0 = FlexiCubes(device=self.device_0)
        self.mesh_extractor_1 = FlexiCubes(device=self.device_1)
        
        self.sdf_bias = -1.0 / self.res
        
        # Construct split dense grids
        self._construct_split_dense_grid()
    
    def _construct_split_dense_grid(self):
        """Construct dense grid split across multiple GPUs."""
        res_v = self.res + 1
        total_verts = res_v ** 3
        
        # Split vertices between GPUs
        split_point = total_verts // 2
        
        print(f"Splitting {total_verts} vertices: GPU0={split_point}, GPU1={total_verts-split_point}")
        
        # GPU 0: First half
        vertsid_0 = torch.arange(split_point, device=self.device_0)
        
        # GPU 1: Second half
        vertsid_1 = torch.arange(split_point, total_verts, device=self.device_1)
        
        # Construct cube corners
        cube_corners_bias = (cube_corners[:, 0] * res_v + cube_corners[:, 1]) * res_v + cube_corners[:, 2]
        
        # Create coordinate grids for cubes that fit within each vertex set
        max_cubes_0 = min(self.res ** 3 // 2, len(vertsid_0))
        max_cubes_1 = min(self.res ** 3 - max_cubes_0, len(vertsid_1))
        
        # GPU 0 setup
        if max_cubes_0 > 0:
            coordsid_0 = vertsid_0[:max_cubes_0]
            self.cube_fx8_0 = (coordsid_0.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(self.device_0))
            self.verts_0 = torch.stack([
                vertsid_0 // (res_v ** 2), 
                (vertsid_0 // res_v) % res_v, 
                vertsid_0 % res_v
            ], dim=1).float()
        else:
            self.cube_fx8_0 = torch.empty(0, 8, device=self.device_0, dtype=torch.long)
            self.verts_0 = torch.empty(0, 3, device=self.device_0)
        
        # GPU 1 setup
        if max_cubes_1 > 0:
            coordsid_1 = vertsid_1[:max_cubes_1]
            self.cube_fx8_1 = (coordsid_1.unsqueeze(1) + cube_corners_bias.unsqueeze(0).to(self.device_1))
            self.verts_1 = torch.stack([
                vertsid_1 // (res_v ** 2), 
                (vertsid_1 // res_v) % res_v, 
                vertsid_1 % res_v
            ], dim=1).float()
        else:
            self.cube_fx8_1 = torch.empty(0, 8, device=self.device_1, dtype=torch.long)
            self.verts_1 = torch.empty(0, 3, device=self.device_1)
    
    def _calc_layout(self):
        LAYOUTS = {
            'sdf': {'shape': (8, 1), 'size': 8},
            'deform': {'shape': (8, 3), 'size': 8 * 3},
            'weights': {'shape': (21,), 'size': 21}
        }
        if self.use_color:
            LAYOUTS['color'] = {'shape': (8, 6,), 'size': 8 * 6}
        self.layouts = edict(LAYOUTS)
        start = 0
        for k, v in self.layouts.items():
            v['range'] = (start, start + v['size'])
            start += v['size']
        self.feats_channels = start
        
    def get_layout(self, feats: torch.Tensor, name: str):
        if name not in self.layouts:
            return None
        return feats[:, self.layouts[name]['range'][0]:self.layouts[name]['range'][1]].reshape(-1, *self.layouts[name]['shape'])
    
    def __call__(self, cubefeats: SparseTensor, training=False):
        """
        Generates a mesh based on the specified sparse voxel structures.
        """
        if self.enable_multi_gpu and torch.cuda.device_count() > 1:
            return self._call_multi_gpu(cubefeats, training)
        else:
            return self._call_single_gpu(cubefeats, training)
    
    def _call_single_gpu(self, cubefeats: SparseTensor, training=False):
        """Original single GPU implementation."""
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        v_attrs = [sdf, deform, color] if self.use_color else [sdf, deform]
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        if self.use_color:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
            
        x_nx3 = get_defomed_verts(self.reg_v, deform_d, self.res)
        
        vertices, faces, L_dev, colors = self.mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=self.reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training)
        
        mesh = MeshExtractResult(vertices=vertices, faces=faces, vertex_attrs=colors, res=self.res)
        if training:
            if mesh.success:
                reg_loss += L_dev.mean() * 0.5
            reg_loss += (weights[:,:20]).abs().mean() * 0.2
            mesh.reg_loss = reg_loss
            mesh.tsdf_v = get_defomed_verts(v_pos, v_attrs[:, 1:4], self.res)
            mesh.tsdf_s = v_attrs[:, 0]
        return mesh
    
    def _call_multi_gpu(self, cubefeats: SparseTensor, training=False):
        """Multi-GPU implementation with model parallelism."""
        coords = cubefeats.coords[:, 1:]
        feats = cubefeats.feats
        
        # Extract feature components
        sdf, deform, color, weights = [self.get_layout(feats, name) for name in ['sdf', 'deform', 'color', 'weights']]
        sdf += self.sdf_bias
        
        # Split data between GPUs based on spatial coordinates
        coords_cpu = coords.cpu()
        if len(coords_cpu) == 0:
            # Handle empty case
            return MeshExtractResult(
                vertices=torch.empty(0, 3, device=self.device_0),
                faces=torch.empty(0, 3, device=self.device_0),
                vertex_attrs=None,
                res=self.res
            )
        
        # Split along the first spatial dimension
        mid_coord = coords_cpu[:, 0].max().item() // 2
        mask_0 = coords_cpu[:, 0] <= mid_coord
        mask_1 = ~mask_0
        
        # Process on both GPUs in parallel
        results = []
        
        # GPU 0 processing
        if mask_0.sum() > 0:
            result_0 = self._process_on_device(
                coords[mask_0], sdf[mask_0], deform[mask_0], 
                color[mask_0] if self.use_color else None, 
                weights[mask_0], self.device_0, 0, training
            )
            results.append(result_0)
        
        # GPU 1 processing  
        if mask_1.sum() > 0:
            result_1 = self._process_on_device(
                coords[mask_1], sdf[mask_1], deform[mask_1],
                color[mask_1] if self.use_color else None,
                weights[mask_1], self.device_1, 1, training
            )
            results.append(result_1)
        
        # Combine results
        return self._combine_results(results, training)
    
    def _process_on_device(self, coords, sdf, deform, color, weights, device, gpu_id, training):
        """Process features on a specific GPU."""
        if len(coords) == 0:
            return None
        
        # Move data to target device
        coords = coords.to(device)
        sdf = sdf.to(device)
        deform = deform.to(device)
        weights = weights.to(device)
        if color is not None:
            color = color.to(device)
        
        # Prepare vertex attributes
        v_attrs = [sdf, deform]
        if self.use_color and color is not None:
            v_attrs.append(color)
        
        # Process vertex attributes
        v_pos, v_attrs, reg_loss = sparse_cube2verts(coords, torch.cat(v_attrs, dim=-1), training=training)
        v_attrs_d = get_dense_attrs(v_pos, v_attrs, res=self.res+1, sdf_init=True)
        weights_d = get_dense_attrs(coords, weights, res=self.res, sdf_init=False)
        
        if self.use_color and color is not None:
            sdf_d, deform_d, colors_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4], v_attrs_d[..., 4:]
        else:
            sdf_d, deform_d = v_attrs_d[..., 0], v_attrs_d[..., 1:4]
            colors_d = None
        
        # Get appropriate grid for this GPU
        if gpu_id == 0:
            reg_v = self.verts_0
            reg_c = self.cube_fx8_0
            mesh_extractor = self.mesh_extractor_0
        else:
            reg_v = self.verts_1
            reg_c = self.cube_fx8_1
            mesh_extractor = self.mesh_extractor_1
        
        # Deform vertices
        x_nx3 = get_defomed_verts(reg_v, deform_d, self.res)
        
        # Extract mesh
        vertices, faces, L_dev, colors = mesh_extractor(
            voxelgrid_vertices=x_nx3,
            scalar_field=sdf_d,
            cube_idx=reg_c,
            resolution=self.res,
            beta=weights_d[:, :12],
            alpha=weights_d[:, 12:20],
            gamma_f=weights_d[:, 20],
            voxelgrid_colors=colors_d,
            training=training
        )
        
        return {
            'vertices': vertices,
            'faces': faces,
            'colors': colors,
            'L_dev': L_dev,
            'reg_loss': reg_loss,
            'v_pos': v_pos,
            'v_attrs': v_attrs,
            'weights': weights
        }
    
    def _combine_results(self, results, training):
        """Combine results from multiple GPUs."""
        if not results:
            return MeshExtractResult(
                vertices=torch.empty(0, 3, device=self.device_0),
                faces=torch.empty(0, 3, device=self.device_0),
                vertex_attrs=None,
                res=self.res
            )
        
        # Move all results to GPU 0 and combine
        combined_vertices = []
        combined_faces = []
        combined_colors = []
        total_reg_loss = 0.0
        vertex_offset = 0
        
        for result in results:
            if result is None:
                continue
            
            # Move to GPU 0
            vertices = result['vertices'].to(self.device_0)
            faces = result['faces'].to(self.device_0) + vertex_offset
            
            combined_vertices.append(vertices)
            combined_faces.append(faces)
            
            if result['colors'] is not None:
                combined_colors.append(result['colors'].to(self.device_0))
            
            total_reg_loss += result['reg_loss']
            vertex_offset += vertices.shape[0]
        
        # Concatenate results
        if combined_vertices:
            final_vertices = torch.cat(combined_vertices, dim=0)
            final_faces = torch.cat(combined_faces, dim=0)
            final_colors = torch.cat(combined_colors, dim=0) if combined_colors else None
        else:
            final_vertices = torch.empty(0, 3, device=self.device_0)
            final_faces = torch.empty(0, 3, device=self.device_0)
            final_colors = None
        
        mesh = MeshExtractResult(
            vertices=final_vertices,
            faces=final_faces,
            vertex_attrs=final_colors,
            res=self.res
        )
        
        if training:
            mesh.reg_loss = total_reg_loss
            # Note: tsdf_v and tsdf_s would need special handling for multi-GPU
        
        print(f"✅ Multi-GPU mesh: {final_vertices.shape[0]} vertices, {final_faces.shape[0]} faces")
        return mesh