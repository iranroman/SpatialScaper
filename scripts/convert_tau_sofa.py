import os
from room_scaper import sofa_utils

def create_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name):
    for room_idx in range(9):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        rirs, source_pos, mic_pos, room = sofa_utils.load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt=aud_fmt)
        
        filepath = os.path.join(sofa_db_dir, aud_fmt, f'{room}.sofa')
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

# Call the function for both 'foa' and 'mic'
tau_db_dir = 'TAU_DB/TAU-SRIR_DB'
sofa_db_dir = 'TAU_DB/TAU_SRIR_DB_SOFA'
db_name = "TAU-SRIR-DB-SOFA"
for aud_fmt in ['foa', 'mic']:
    print(f"Starting .sofa creation for {aud_fmt} format.")
    create_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name)
    print(f"Finished .sofa creation for {aud_fmt} format.")
