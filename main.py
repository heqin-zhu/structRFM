from datetime import datetime

from src.SgRFM.pretrain import run_pretrain


if __name__ == '__main__':
    print(f'Time: {datetime.now()}')
    run_pretrain()
