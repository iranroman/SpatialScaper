import os
from room_scaper import sofa_utils, tau_loading

rooms = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203', 'tb103', 'tc352']


def create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name):
    db_dir = os.path.join(sofa_db_dir, aud_fmt)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    for room_idx in range(9):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        rirs, source_pos, mic_pos, room = sofa_utils.load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt=aud_fmt)
        
        filepath = os.path.join(db_dir, f'{room}.sofa')
        comment = f"SOFA conversion of {room} from TAU-SRIR-DB"
        
        print(f"Creating .sofa file for {aud_fmt}, Room: {room} (Progress: {room_idx + 1}/9)")
        
        # Create .sofa files with flattened rirs/paths + metadata
        sofa_utils.create_srir_sofa(
            filepath,
            rirs,
            source_pos,
            mic_pos,
            db_name=db_name,
            room_name=room,
            listener_name=aud_fmt,
            sr=24000,
            comment=comment
        )
        
def create_per_traj_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name):
    db_dir = os.path.join(sofa_db_dir, aud_fmt)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    for room_idx in range(9):
        room = rooms[room_idx]
        room_dir = os.path.join(db_dir,room)
        if not os.path.exists(room_dir):
            os.makedirs(room_dir)
        n_traj = tau_loading.check_n_traj(tau_db_dir, room_idx)
        for traj_idx in range(n_traj):
            # Load flattened (and flipped) rirs/paths from TAU-SRIR database
            rirs, source_pos, mic_pos, room = sofa_utils.load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt=aud_fmt, traj=traj_idx)

            filepath = os.path.join(room_dir, f'{room}_t{traj_idx}.sofa')
            comment = f"SOFA conversion of {room}, trajectory {traj_idx} from TAU-SRIR-DB"

            print(f"Creating .sofa file for {aud_fmt}, Room: {room}, trajectory {traj_idx} (Progress: {room_idx + 1}/9, {traj_idx}/{n_traj})")

            # Create .sofa files with flattened rirs/paths + metadata
            sofa_utils.create_srir_sofa(
                filepath,
                rirs,
                source_pos,
                mic_pos,
                db_name=db_name,
                room_name=room,
                listener_name=aud_fmt,
                sr=24000,
                comment=comment
            )
        


# Call the function for both 'foa' and 'mic'
tau_db_dir = 'TAU_DB/TAU-SRIR_DB'
sofa_db_dir = 'TAU_DB/TAU_SRIR_DB_SOFA'
#tau_db_dir = '/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB/TAU-SRIR_DB'
#sofa_db_dir = '/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB_SOFA'
db_name = "TAU-SRIR-DB-SOFA"
for aud_fmt in ['foa', 'mic']:
    print(f"Starting .sofa creation for {aud_fmt} format, per traj.")
    create_per_traj_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name)
    print(f"Finished .sofa creation for {aud_fmt} format, per traj.")
    
    print(f"Starting .sofa creation for {aud_fmt} format.")
    create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name)
    print(f"Finished .sofa creation for {aud_fmt} format.")
