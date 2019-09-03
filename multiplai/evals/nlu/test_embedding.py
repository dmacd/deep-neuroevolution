
import pytest
import multiplai.evals.nlu.embedding as embedding
import numpy as np
from mxnet import nd

TEST_HELLO_WORLD_EMBEDDING = np.array([[ 3.9567e-01,  2.1454e-01, -3.5389e-02,
                               -2.4299e-01,
                         -9.5645e-02,
         7.0196e-01, -2.2217e-01,  5.9566e-01,  3.5400e-01,  1.1223e-01,
         1.7507e-01, -1.8420e-01, -4.5743e-01,  2.0431e-01, -6.8720e-02,
        -7.5822e-03, -3.1745e-02, -1.2260e-01, -8.4278e-02, -4.0954e-01,
         1.5536e-01, -1.0237e-01,  1.6776e-01,  7.2908e-02, -3.3974e-01,
         4.0964e-01,  3.1588e-01, -3.0134e-01,  4.5805e-01,  5.0324e-02,
        -5.4599e-01, -4.3832e-02, -1.1785e-01,  2.2829e-01,  1.9781e-01,
         2.7429e-01,  1.5484e-01, -1.6361e-02,  5.6247e-01, -2.8905e-02,
         3.2125e-01, -2.2172e-01, -1.8451e-01, -9.5252e-02, -3.7475e-01,
        -3.0096e-03, -1.7133e-01, -2.4177e-02, -4.2394e-02,  5.5292e-02,
        -3.6133e-01,  7.9608e-02, -7.3816e-02, -1.3945e-01, -5.7920e-02,
         6.0948e-03,  3.1616e-01, -6.9531e-02, -4.1851e-01, -7.8901e-02,
        -1.9404e-01, -1.6711e-02, -4.2909e-01, -3.2553e-02, -4.0582e-01,
        -1.8679e-01,  2.3638e-02, -3.6106e-01, -1.1680e-04,  6.6403e-02,
         1.0071e-01,  2.1626e-01, -1.2835e-01,  3.6103e-01, -1.3715e-01,
        -6.1190e-02, -3.8188e-01, -2.8119e-01,  3.6750e-01,  7.1560e-03,
         5.0234e-01, -1.1317e-01, -1.5920e-01,  4.4633e-02, -3.0035e-01,
        -2.1490e-01,  2.3501e-01, -2.2151e-01,  9.2836e-02, -8.1031e-01,
        -4.6148e-01,  7.8923e-02, -4.9757e-01, -2.5092e-02,  4.4697e-01,
         2.4795e-01, -7.1856e-01, -1.5004e-01,  1.1904e-01, -2.0064e-01,
         1.3000e-01, -6.2002e-02, -1.7859e-02, -6.6152e-03, -4.8471e-01,
         8.1129e-02, -5.3102e-01,  4.1571e-02, -4.6091e-02, -2.0755e-01,
        -1.4421e-01, -2.5514e-01,  6.7258e-02, -1.0041e-01, -2.2771e-01,
        -2.9983e-01,  1.5760e-01,  7.8602e-01, -4.7287e-01,  7.8859e-02,
         6.9349e-02,  4.8157e-01, -3.8714e-02,  6.4797e-02, -4.8402e-01,
        -8.2103e-01,  1.1725e-01,  4.1851e-02,  5.4920e-02, -1.8430e-01,
         1.1429e-01,  1.7205e-01, -7.3599e-02, -1.6165e-02, -3.4740e-01,
        -6.0199e-02,  2.5808e-01, -1.3113e-01, -1.6971e-01, -4.2851e-01,
        -6.5230e-01,  1.8635e-01, -2.3473e-01,  1.2852e-01,  1.9721e-01,
         8.0693e-02, -5.2418e-01, -2.2415e-01,  1.1740e-01, -7.6664e-02,
        -2.2182e-01,  8.9106e-02,  2.0090e-01,  1.9626e-02,  4.0090e-01,
         1.5189e-01, -6.7135e-02,  2.2292e-02,  1.2248e-01, -1.7770e-01,
        -2.3174e-01, -3.5582e-01,  1.5086e-01, -8.8843e-02,  3.0459e-02,
         4.1640e-02,  5.2011e-02, -4.3609e-01,  5.0947e-01,  1.8445e-01,
        -2.2596e-01, -9.7855e-02, -4.6621e-01,  7.7472e-02,  2.0312e-02,
         1.0437e-02, -1.9767e-01, -1.4682e-01,  3.9875e-01, -1.4261e-01,
        -6.3095e-01,  3.9198e-02, -3.2454e-01,  2.7128e-01,  2.6706e-01,
         1.8585e-02,  2.1183e-01, -7.2323e-02,  1.2289e-01,  6.4784e-01,
         1.4635e-01,  9.6362e-02,  2.5686e-01, -7.2680e-02, -3.7782e-01,
        -4.3519e-01, -8.7646e-01, -1.9399e-01,  6.2858e-02, -3.0186e-01,
         3.5454e-01,  2.6528e-01,  2.1853e-01,  1.5688e-01, -5.0348e-01,
         1.7768e-01,  7.4341e-02,  4.5868e-01, -3.1830e-01, -2.0047e-01,
        -1.6847e-01, -9.8182e-02,  4.9339e-01, -5.0095e-01, -1.9910e-01,
         2.1468e-01, -1.8846e-01, -3.2755e-01, -3.5421e-01,  3.5366e-01,
        -2.9484e-01, -2.1658e-02, -4.8068e-01,  1.1578e-01,  1.6799e-01,
        -1.2026e-01, -8.6954e-03,  2.6344e-01, -4.3505e-02,  1.5306e-01,
         5.8245e-02,  4.8804e-01, -3.1821e-01,  1.6461e-01, -3.4359e-01,
         1.7962e-01, -3.1692e-01,  1.0989e-02,  1.4989e-01, -3.1745e-02,
         2.9266e-01, -2.4258e-03, -2.2247e-01,  2.4630e-01,  6.7556e-01,
        -1.7419e-01,  6.9367e-01, -2.1804e-01,  3.2332e-02,  3.6521e-02,
        -1.8767e-01,  5.1153e-01, -6.0152e-01, -8.0548e-02, -3.0559e-02,
        -1.4123e-02, -1.3207e-01,  3.6317e-01, -4.3494e-01,  8.0458e-02,
        -3.2099e-01, -4.3120e-01,  2.6166e-01,  4.2963e-01, -7.9263e-02,
        -1.9196e-01,  1.6546e-01,  9.6910e-02, -1.6433e-01, -7.8967e-01,
        -3.0059e-02, -1.5591e-01,  3.6001e-01, -2.7975e-02,  1.4566e-01,
         3.1766e-01, -7.3518e-02,  5.6064e-01,  3.4178e-01, -3.0879e-01,
         2.6112e-01, -1.3669e-01, -1.2612e-01, -6.3995e-01,  5.6239e-02,
         3.3859e-01,  4.1397e-02,  5.4060e-02,  1.6268e-01,  5.1144e-01,
        -6.6267e-02, -2.6571e-01, -5.1054e-01, -4.2417e-01,  7.4714e-01,
         4.0900e-02, -7.5418e-01, -3.1443e-01,  2.4018e-02, -7.6101e-02],
       [ 1.0444e-01, -1.0858e-01,  2.7212e-01,  1.3299e-01, -3.3165e-01,
         2.4310e-01, -6.1464e-02,  2.6072e-01,  3.4468e-01,  2.4314e-02,
         2.9228e-01, -8.3887e-02, -2.3797e-01,  3.2260e-01, -9.0141e-02,
         3.7859e-01,  3.3501e-01,  1.1622e-03, -2.7804e-01, -6.6614e-03,
         2.4534e-02, -1.1212e-01,  3.4948e-02, -4.0253e-05,  6.0458e-02,
         3.6715e-02,  2.1343e-01, -3.7608e-01,  4.1616e-02,  5.2514e-02,
        -2.0079e-02,  4.8246e-02,  2.1048e-01,  1.6611e-01, -1.8211e-01,
         2.4476e-01,  2.1202e-02,  6.7158e-02,  2.1180e-02, -2.4979e-01,
        -2.0303e-01,  1.7363e-01, -1.2754e-01, -3.4262e-01,  2.1358e-02,
        -1.9503e-01, -7.4986e-02,  1.2362e-01,  5.5324e-02,  1.8145e-01,
         4.0675e-02, -9.8244e-02, -1.3708e-01,  5.1536e-02, -4.1870e-01,
         1.4260e-01,  3.6313e-01, -5.8779e-02, -2.7624e-01,  3.6986e-02,
         1.3877e-01,  5.3143e-01, -1.6731e-01, -1.1746e-01, -8.4201e-02,
        -6.2219e-02, -5.8623e-01, -1.4632e-02,  2.3573e-01, -3.8330e-01,
         1.4662e-01, -1.8183e-01, -1.5338e-01, -3.0602e-01,  3.5237e-01,
         1.1277e-01,  4.5678e-01,  6.5883e-02,  5.7273e-02,  2.5625e-01,
        -1.9320e-01,  1.8087e-02,  6.9407e-02,  1.7856e-01,  3.4650e-02,
        -2.2421e-02,  2.6947e-01,  4.2207e-01, -3.0752e-01, -5.2701e-01,
         1.8882e-01, -1.4869e-01,  7.8663e-02,  2.3430e-01,  1.1904e-01,
         5.9574e-02, -9.8482e-02,  1.1064e-01,  1.3252e-01,  2.1671e-01,
         2.8438e-01, -2.4744e-01,  6.0942e-02,  2.5386e-01, -1.2168e-01,
         2.9463e-01,  2.7697e-01,  1.2252e-01, -2.2059e-01, -1.4248e-02,
         9.0809e-03,  1.3505e-01,  1.0437e-01, -3.7177e-01, -1.8320e-01,
        -4.3265e-01, -4.6595e-02,  3.4300e-02, -1.5024e-01, -2.1485e-01,
        -7.3676e-02, -3.3719e-01, -2.1505e-01,  2.5815e-01, -4.3679e-01,
        -2.3173e-01,  9.6714e-02, -2.2807e-01, -1.5066e-02,  6.9735e-02,
         5.6453e-02, -9.8034e-02, -6.8961e-02, -1.8716e-01,  2.7394e-01,
         2.1957e-01,  4.6782e-01, -3.7712e-01, -1.9008e-02, -2.3368e-01,
        -3.5259e-01,  5.9267e-02,  4.8267e-01,  1.2267e-01,  5.0345e-01,
        -9.6354e-02, -2.3975e-01, -2.5524e-01, -1.1917e-01,  1.1379e-01,
        -9.8741e-03, -1.2126e-01,  1.2464e-01,  3.8297e-02, -2.2239e-02,
        -3.5653e-02,  3.8932e-02,  5.9508e-02,  6.1127e-02,  2.3562e-01,
        -1.4115e-01,  2.2822e-01, -3.0941e-01,  2.5656e-01,  1.3342e-01,
         1.0079e-01, -1.1905e-02, -1.7844e-01,  2.2918e-01,  3.3880e-02,
         2.5223e-01, -1.3928e-01, -5.3243e-01,  8.0809e-02, -3.0140e-01,
         1.0194e-01,  7.8433e-02, -2.5237e-01,  2.4291e-01, -2.4109e-01,
        -1.0122e-01,  7.3230e-02,  2.2168e-01,  5.0729e-01,  1.6570e-02,
         1.2802e-01, -2.0080e-02,  1.9922e-01, -1.1023e-01,  6.1828e-02,
        -4.1450e-01, -3.1878e-01,  2.0349e-01,  2.4047e-01, -3.4013e-02,
        -1.7436e-01, -4.2902e-01, -1.1901e-01, -1.6826e-01, -3.7574e-01,
        -1.7323e-01,  3.2867e-01, -1.6948e-01,  1.8518e-01,  3.1439e-02,
        -2.7551e-01, -1.1895e-01, -6.5302e-02, -2.1983e-01, -3.6821e-01,
        -3.8802e-01, -1.5850e-01,  1.9178e-01, -1.0980e-01, -2.0218e-01,
        -3.3861e-02,  1.9018e-01, -3.8941e-01, -1.3294e-01,  2.2738e-01,
         1.8017e-01,  1.5887e-02,  7.5228e-02, -3.3344e-02, -3.5941e-01,
         3.6348e-01, -4.4265e-02,  9.9925e-02, -1.3576e-02,  2.8717e-01,
         2.5186e-01,  2.6647e-01, -3.6377e-01, -3.8433e-01, -5.9472e-02,
        -1.9699e-01,  2.3070e-01, -3.3024e-02, -1.7652e-01,  1.6155e-01,
         1.9225e-01,  3.3793e-02, -4.0751e-02, -3.5201e-01, -1.0364e-01,
        -3.4577e-01, -1.3184e-01,  6.8251e-02, -2.9135e-01,  1.5851e-01,
         1.3989e-01,  1.5149e-02, -3.1488e-01, -1.9117e-01, -7.4099e-02,
        -9.6962e-02, -2.7640e-02,  3.1426e-01, -2.1744e-01,  4.9866e-01,
        -2.7552e-01,  9.2107e-02,  2.5650e-01,  6.4093e-02, -2.1806e-02,
         1.0756e-01,  7.8579e-02,  1.3654e-01, -8.9795e-02,  1.4376e-01,
        -2.7066e-01,  1.1344e-01,  2.3630e-01,  1.8995e-01,  7.4664e-04,
         7.4619e-02,  1.7155e-01,  6.3847e-02,  2.3327e-01,  3.8975e-03,
         1.0732e-01, -1.1305e-01, -8.3866e-02, -1.0916e-01,  1.0301e-01,
         8.0568e-02,  3.4770e-02,  7.1380e-02, -1.8292e-01,  2.6494e-01,
         1.4107e-02, -4.8173e-02, -2.1358e-01,  2.7687e-01,  7.9292e-02,
         1.8825e-01, -3.7350e-01,  5.6731e-02,  5.6018e-01,  2.9019e-02]],
      dtype=np.float32)

def test_embedding():
    e = embedding.get_embedding_for_text(" hello world \n hello nice world \n "
                                       "hi world \n")


    vecs = e.get_vecs_by_tokens(['hello', 'world']).asnumpy()

    assert np.allclose(vecs, TEST_HELLO_WORLD_EMBEDDING)

    layer = embedding.layer_for_embedding(e)

    assert np.allclose(layer(nd.array([2,1])).asnumpy(),
                       TEST_HELLO_WORLD_EMBEDDING)


