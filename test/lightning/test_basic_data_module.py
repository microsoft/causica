import pandas as pd
import torch

from causica.datasets.causica_dataset_format import Variable
from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule


def test_basic_data_module():
    """Test Basic Data Module functionality"""
    df = pd.DataFrame(
        {
            "x0_0": {
                0: -0.5569139122962952,
                1: 0.5148046016693115,
                2: -0.29866957664489746,
                3: -0.9968659877777101,
                4: 2.916576147079468,
                5: -0.05121749639511108,
                6: 0.4290653467178345,
                7: -0.7042781710624695,
                8: -0.06559652090072632,
                9: -0.9346157908439635,
            },
            "x0_1": {
                0: 0.06510090827941895,
                1: 1.4334404468536377,
                2: -0.9826208949089049,
                3: -0.2685583829879761,
                4: 1.1215617656707764,
                5: 0.7001379728317261,
                6: -0.1907709240913391,
                7: 0.49415969848632807,
                8: -0.2164681553840637,
                9: 0.5424903631210327,
            },
            "x1_0": {
                0: 0.008327394723892214,
                1: -0.3806004524230957,
                2: -0.2750669717788696,
                3: -0.2773442268371582,
                4: 4.198153972625732,
                5: -0.7430528402328491,
                6: -0.07149887084960938,
                7: 0.5431607961654663,
                8: 0.2636866867542267,
                9: -0.5033876895904541,
            },
            "x1_1": {
                0: 0.33135163784027094,
                1: 0.03788989782333375,
                2: -1.343551516532898,
                3: -0.39070096611976624,
                4: 0.04580897092819214,
                5: -0.41362786293029785,
                6: -0.6606348752975464,
                7: 0.021188572049140927,
                8: -0.8007687330245972,
                9: 0.4804639220237732,
            },
        }
    )
    variables_list = [
        Variable(
            group_name="x0",
            lower=-0.9997121691703796,
            name="x0_0",
            upper=9.77190113067627,
        ),
        Variable(
            group_name="x0",
            lower=-0.9997121691703796,
            name="x0_1",
            upper=9.77190113067627,
        ),
        Variable(
            group_name="x1",
            lower=-1.3532483577728271,
            name="x1_0",
            upper=6.881451606750488,
        ),
        Variable(
            group_name="x1",
            lower=-1.3532483577728271,
            name="x1_1",
            upper=6.881451606750488,
        ),
    ]
    data_module = BasicDECIDataModule(df, variables_list, batch_size=2)

    assert list(data_module.train_dataloader())[0].batch_size == torch.Size([2])
    assert data_module.dataset_train.batch_size == torch.Size([10])
    assert data_module.dataset_train["x0"].shape == torch.Size([10, 2])
