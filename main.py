from datetime import datetime

from src.NucleoLM.pretrain import run_pretrain


if __name__ == '__main__':
    print(f'Time: {datetime.now()}')
    run_pretrain()
