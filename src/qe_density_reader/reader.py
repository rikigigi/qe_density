import h5py
import numpy as np
import math
import k3d
import k3d.transform

class Density:
    def __init__(self, fname: str):
        '''
        returns a numpy array given the hdf5 file written by the routine
        
        https://gitlab.com/rikigigi/q-e/-/blob/write_rhor_hdf5/Modules/io_base.f90?ref_type=heads#L995
        '''
        rho=h5py.File(fname,'r')
        self.hdf5file=rho
        rho_data=rho.get('rho_of_r_data')
        rho_data_np = np.array(rho_data)
        rho_index_data_np = np.array(rho.get('rho_of_r_index_size'))
        rho_cell=rho_data.attrs['cell_vectors']
        alat = rho_data.attrs['alat']
        self.rho_shape=rho_data.attrs['grid_dimensions']
        rho_full = np.zeros(self.rho_shape[::1])
        atoms_positions = rho_data.attrs['tau']
        atoms_types = rho_data.attrs['ityp']
        #NOTE: in qe I have the option to not save density values lower than a threshold to save some disk space.
        count=0
        for idx,size in zip(rho_index_data_np[0::2],rho_index_data_np[1::2]):
            # NB: the index from the file are fortran one, so the first element is 1.
            # But in python indexes starts from 0, so we have the "-1"
            rho_full.flat[idx-1:idx-1+size]=rho_data_np[count:count+size]
            count+=size
        
        print(f'''fname={fname}:
     items: {list(rho.items())}
     attrs: {list(rho_data.attrs.items())}
     compress_ratio: {(rho_data_np.size+rho_index_data_np.size)/rho_full.size}
               ''')
        print (rho_index_data_np)
        self.V = np.linalg.det(rho_cell*alat)
        self.nr=np.array(rho_full.shape).prod()
        #save variables into current instance
        self.hdf_file=rho
        self.rho=rho_full
        self.type_idx=rho_data.attrs['ityp']-1
        self.atomic_charges=rho_data.attrs['zv'][self.type_idx]
        self.atoms_positions=atoms_positions*alat
        self.cell=rho_cell*alat
        self.alat = alat
        self.rho_data=rho_data
        self.dipole_total=None

    def idx2r(self,*args):
        ijk=np.array(args)/self.rho_shape
        #cell vectors are cell[0],cell[1],cell[2]
        return ijk@self.cell
        
    def dipole(self):
        '''
          dipole = \sum_{ijk} r_{ijk} \rho_{ijk} dV
          warning: the molecule must be in the center of the cell,
          and the charge must be zero near the boundaries or the calculation is ill-defined
        '''
        if np.any(self.cell != self.cell.transpose()):
            raise NotImplemented('Only orthogonal cells are implemented')
        ang=0.52917721 #bohr radius
        e_ang2D=0.2081943 #https://en.wikipedia.org/wiki/Debye
        r0=0.0
        #atomic dipole
        #atomic dipole in electrons charges times Angstrom
        atomic_dipole=np.tensordot(self.atoms_positions*ang-r0,self.atomic_charges,axes=((0,),(0,)))
        print('dipole nuclei, D',atomic_dipole*e_ang2D)

        #electronic dipole is equivalent to the following:
        #dip=0
        #dV=np.linalg.det(cell)/nr
        #for i in range(rho_full.shape[0]):
        #    for j in range(rho_full.shape[1]):
        #        for k in range(rho_full.shape[2]):
        #            r = idx2r(i,j,k)
        #            dip+=r*rho_full[i,j,k]*dV
        
        dV = self.V/self.nr
        #index grid
        grid = np.array(np.meshgrid(np.arange(0,self.rho_shape[0]),np.arange(0,self.rho_shape[1]),np.arange(0,self.rho_shape[2]),indexing='ij'))
        #coord grid
        coords = np.tensordot(grid/self.rho_shape[:,None,None,None],self.cell, axes=(0,1))*ang - r0
        el_dipole=np.sum((coords)*self.rho[...,None], axis=(0,1,2))*dV
        #dipole
        print('dipole electrons, D', el_dipole*e_ang2D,dV,self.V,self.nr)
        dipole_total=(-el_dipole + atomic_dipole)*e_ang2D
        print(dipole_total)
        self.dipole_total = dipole_total
        return np.linalg.norm(dipole_total)        
        
    def display(self):
        rot=np.identity(4)
        trans=np.identity(4)
        rot[:3,:3]=self.cell.transpose()
        trans[:3,3]=+np.ones(3)/2        
        transform = k3d.transform(custom_matrix=rot@trans)
        rho_full_draw=k3d.volume(self.rho,alpha_coef=15)
        transform.add_drawable(rho_full_draw)
        transform.parent_updated()
        rho_full_draw.transform=transform
        p=k3d.plot()
        p+=k3d.points(self.atoms_positions,point_size=0.5)
        p+=rho_full_draw
        return p.display()
        
    def write_compressed_hdf5(self,output_file: str):
    # Open the input HDF5 file in read 
        with h5py.File(output_file, 'w') as f_out:
            # Iterate over all groups and datasets in the input file
            def copy_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    # Create a new dataset in the output file with the same name, shape, and datatype
                    dset_out = f_out.create_dataset(name, shape=obj.shape, dtype=obj.dtype,
                                                    compression="gzip", compression_opts=9,
                                                    shuffle=True)
                    # Copy the data from the input dataset to the output dataset with compression
                    dset_out[...] = obj[...]
                    for attr_name, attr_value in obj.attrs.items():
                        dset_out.attrs[attr_name] = attr_value
            self.hdf5file.visititems(copy_dataset)
            if self.dipole_total is not None:
                try:
                    dset=f_out.create_dataset('dipole_total',data=self.dipole_total)
                except Exception as e:
                    print(e)
