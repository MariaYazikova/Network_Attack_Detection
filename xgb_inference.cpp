#include <xgboost/c_api.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

int main()
{
    const char *classes[] = {
        "BENIGN",
        "DoS Slowloris",
        "DoS Slowhttptest",
        "DoS Hulk",
        "DoS GoldenEye",
        "Infiltration - Portscan",
        "Portscan",
        "DDoS"};

    std::ifstream file("inputs/inputs.csv");
    std::string line;
    std::getline(file, line);
    std::vector<float> all_data;
    size_t n_rows = 0;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ','))
        {
            all_data.push_back(std::stof(val));
        }
        n_rows++;
    }
    file.close();

    if (n_rows == 0)
    {
        std::cerr << "No data in CSV\n";
        return 1;
    }

    size_t n_features = all_data.size() / n_rows;

    DMatrixHandle dmat;
    XGDMatrixCreateFromMat(all_data.data(), (bst_ulong)n_rows, (bst_ulong)n_features, -1, &dmat);

    BoosterHandle booster;
    XGBoosterCreate(&dmat, 1, &booster);

    const char *model_path = "models/saved/xgb_model.json";
    XGBoosterLoadModel(booster, model_path);

    bst_ulong out_len;
    const float *out_result;
    int training = 0;
    XGBoosterPredict(booster, dmat, 0, 0, training, &out_len, &out_result);

    for (size_t i = 0; i < n_rows; ++i)
    {
        int idx = static_cast<int>(out_result[i]);
        std::cout << "Row " << i << ": Predicted class = " << idx
                  << " (" << classes[idx] << ")" << std::endl;
    }

    XGBoosterFree(booster);
    XGDMatrixFree(dmat);

    return 0;
}