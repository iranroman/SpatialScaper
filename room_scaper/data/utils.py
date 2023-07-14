#    'tc352','sc203','bomb_shelter','pc226','pb132','se203',
#    'tb103','sa203','gym'
import glob 

PATH_TO_RIRS = 'room_scaper/data/rir_databases'

def ROOM_REGISTRY():
    room_names = glob.glob(PATH_TO_RIRS + '/**/*.pkl')
    print(room_names)
    input()

