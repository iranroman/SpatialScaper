import sofa_utils
import os

tau_db_dir = '/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB/TAU-SRIR_DB'
sofa_db_dir = '/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB_SOFA'
db_name = "TAU-SRIR-DB-SOFA"

for room_idx in range(9):
    for aud_fmt in ['foa','mic']:
        #load flattened (and flipped) rirs/paths from TAU-SRIR database
        rirs, source_pos, mic_pos, room = sofa_utils.load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt=aud_fmt)
        filepath = os.path.join(sofa_db_dir, aud_fmt, room+'.sofa')
        comment = f"SOFA conversion of {room} from TAU-SRIR-DB"
        #create .sofa files with flattened rirs/paths + metadata
        sofa_utils.create_srir_sofa(filepath, rirs, source_pos, mic_pos, db_name=db_name,\
                     room_name=room, listener_name=aud_fmt, sr=24000, comment=comment)