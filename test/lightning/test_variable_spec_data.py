import json

import numpy as np
import pandas as pd
import pytest
import torch

from causica.datasets.causica_dataset_format import Variable, VariablesMetadata
from causica.lightning.data_modules.variable_spec_data import VariableSpecDataModule

DF = pd.DataFrame(
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


VARIABLES_METADATA = VariablesMetadata(
    variables=[
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
    ],
)
ADJACENCY = np.array([[0, 1], [0, 0]], dtype=int)
INTERVENTIONS = {
    "environments": [
        {
            "conditioning_idxs": [],
            "intervention_idxs": [0],
            "effect_idxs": [1],
            "test_data": np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], axis=1).tolist(),
            "reference_data": np.concatenate([np.ones((5, 2)), np.zeros((5, 2))], axis=1).tolist(),
        }
    ],
    "metadata": {"columns_to_nodes": [0, 0, 1, 1]},
}


@pytest.mark.parametrize("normalize", [True, False])
def test_variable_spec_data(tmp_path, normalize):
    """Test Variable Spec Data Module functionality"""
    variable_spec_path = tmp_path / "variables.json"
    with variable_spec_path.open("w") as f:
        json.dump(VARIABLES_METADATA.to_dict(encode_json=True), f)

    train_path = tmp_path / "train.csv"
    with train_path.open("w") as f:
        DF.to_csv(f, index=False, header=False)

    test_path = tmp_path / "test.csv"
    with test_path.open("w") as f:
        DF.to_csv(f, index=False, header=False)

    adjacency_path = tmp_path / "adj_matrix.csv"
    with adjacency_path.open("w") as f:
        np.savetxt(f, ADJACENCY.tolist(), delimiter=",")

    intervention_path = tmp_path / "interventions.json"
    with intervention_path.open("w") as f:
        json.dump(INTERVENTIONS, f)

    if normalize:
        data_module = VariableSpecDataModule(tmp_path, batch_size=2, normalize=True, exclude_normalization=["x1"])
    else:
        data_module = VariableSpecDataModule(tmp_path, batch_size=2)

    data_module.prepare_data()

    assert list(data_module.train_dataloader())[0].batch_size == torch.Size([2])
    assert data_module.dataset_train.batch_size == torch.Size([10])
    assert data_module.dataset_train["x0"].shape == torch.Size([10, 2])

    if normalize:
        torch.testing.assert_close(data_module.dataset_train["x0"].mean(0), torch.zeros(2))
        assert torch.all(data_module.dataset_train["x1"].mean(0) != torch.zeros(2))
    else:
        assert torch.all(data_module.dataset_train["x0"].mean(0) != torch.zeros(2))
