
/* Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. 
*/

#include "framework/core/net/calibrator.h"
#include "framework/utils/data_common.h"

namespace anakin {
/*shrink ref_p into ref_q which has 128 bins*/
template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::init_statistics(int tensor_num) {
    _max_vec.resize(tensor_num);
    _hist_vecs.resize(tensor_num);
    for (int i = 0; i < tensor_num; i++) {
        _max_vec[i] = dtype(0);
        std::vector<int> hist(_bin_num, 0);
        _hist_vecs[i] = hist;
    }   
}
/*shrink ref_p into ref_q which has 128 bins*/
template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::get_ref_q(std::vector<int>& ref_p, std::vector<float>& ref_q) {
    int p_size = ref_p.size();
    int q_size = ref_q.size();
    float step = p_size * 1.0f / q_size;
    for (int i = 0; i < q_size; i++) {
        float start_pos = step * i;
        float end_pos = step * (i + 1);
        int start_pos_i  = floor(start_pos);
        int end_pos_i = floor(end_pos);
        int start_pos_ceil = ceil(start_pos);
        int count = 0;
        for (int pos = start_pos_ceil; pos < end_pos_i; pos++) {
            count += ref_p[pos];
        }
        count += (start_pos_ceil - start_pos) * ref_p[start_pos_i];
        count += (end_pos - end_pos_i) * ref_p[end_pos_i];
        ref_q[i] = count;
    }
}

/*expand ref_q to q which has as many bins as ref_p*/
template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::expand_to_q(std::vector<int>& ref_p, std::vector<float>& ref_q, std::vector<float>& q) {
    dtype expansion_coeff = dtype(q.size()) / dtype(ref_q.size());

    for (int i = 0; i < ref_q.size(); i++) {
        dtype start_pos = i * expansion_coeff;
        dtype end_pos = (i + 1) * expansion_coeff;
        int start_pos_ceil  =  ceil(start_pos);
        int end_pos_ceil = ceil(end_pos);
        int start_pos_floor = floor(start_pos);
        int end_pos_floor = floor(end_pos);
        dtype zero_num = 0;
        for (int j = start_pos_ceil; j < end_pos_floor; j++) {
            if (ref_p[j] == 0) {
               zero_num += 1;
            }
        }
        if (ref_p[start_pos_floor] == 0) {
            zero_num += start_pos_ceil - start_pos;
        }
        if (ref_p[end_pos_floor] == 0) {
            zero_num += end_pos - end_pos_floor;
        }
        dtype dis = expansion_coeff - zero_num;
        if (ref_p[start_pos_floor] != 0) {
            q[start_pos_floor] += (start_pos_ceil - start_pos) / dis * ref_q[i];
        }
        for (int j = start_pos_ceil; j < end_pos_floor; j++) {
            if (ref_p[j] != 0) {
                q[j] += 1.0f / dis * ref_q[i];
            }
        }
        if (ref_p[end_pos_floor] != 0) {
            q[end_pos_floor] += (end_pos - end_pos_floor) / dis * ref_q[i];
        }
    }
}
/*compute The kl  distance between two distributions*/
/*in this part, tensorrt compute the distance of ref_p and q, we compute the distance of hist and q.
 *  *because the length of q is not equal to hist, The last bin of q was given to all the rest */
template<typename Ttype, DataType Dtype>
typename Calibrator<Ttype, Dtype>::dtype  Calibrator<Ttype, Dtype>:: get_kl_divergence(std::vector<int>&ref_p, std::vector<float>& q) {
//dtype Calibrator<Ttype, Dtype>:: get_kl_divergence(std::vector<int>&ref_p, std::vector<int>& q) {
    int sum_p = 0;
    int sum_q = 0;
    for (int i = 0; i < ref_p.size(); i++) {
        sum_p += ref_p[i];
    }
    for (int i = 0; i< q.size(); i++) {
        sum_q += q[i];
    }
    dtype kl = 0;
    for (int i = 0; i < q.size() - 1; i++) {
        dtype p_prob = dtype(ref_p[i]) / sum_p;
        dtype q_prob = dtype(q[i]) / sum_q;
        if (ref_p[i] != 0 && q[i] != 0 ) {
            kl += p_prob * log2(p_prob/q_prob);
        }
    }
    dtype q_prob = dtype(q[q.size() - 1]) / sum_q/ (ref_p.size() - q.size() + 1);
    for (int i = q.size() - 1; i < ref_p.size(); i++) {
        dtype p_prob = dtype(ref_p[i]) / sum_p;
        if (ref_p[i] > 0) {
            kl += p_prob * log2(p_prob/q_prob);
        }
    }
    return kl;
}

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype, DataType Dtype>
int  Calibrator<Ttype, Dtype>::get_batch_data(std::vector<Tensor4dPtr<Ttype, Dtype>> inputs) {
    if (_in_vec.size() == 0) {
        _in_vec.resize(inputs.size());
        int i = 0;
        for (auto input : inputs) {
            _in_vec[i++] = new Tensor<X86, Dtype>(input->valid_shape());
        }
    }
    int num = _batch_stream->get_batch_data(_in_vec);
    if (num > 0) {
        int i = 0;
        for (auto input : inputs) {
            input->reshape(_in_vec[i]->valid_shape());
            input->copy_from((*_in_vec[i]));
            i++;
        }
    }
    return num;
}


template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::read_calibrator() {
    std::ifstream ifs(_calibrator_table_file);
    CHECK(ifs.is_open()) << _calibrator_table_file << "cannot be opened";
    char buf[200];
    while(true) {
        ifs.getline(buf, 200);
        std::string str = buf;
        std::vector<std::string> delimiter = {" "};
        std::vector<std::string> str_vec = string_split(str, delimiter);
        _scale_map.insert(std::pair<std::string, dtype>(str_vec[0], dtype(atof(str_vec[1].c_str()))));
    }
}

template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::write_calibrator() {
    std::ofstream ofs(_calibrator_table_file);
    CHECK(ofs.is_open()) << _calibrator_table_file << "cannot be opened";
    char buf[200];
    typename std::map<std::string, typename Calibrator<Ttype, Dtype>::dtype>::iterator it;
    for (it = _scale_map.begin(); it != _scale_map.end(); ++it) {
        int n = sprintf(buf, "%s %f\n", it->first.c_str(), float(it->second));
        ofs.write(buf, n);
    }
    ofs.close();
}
template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::reset_data_stream() {

    return _batch_stream->reset();
}

template<typename Ttype, DataType Dtype>
typename Calibrator<Ttype, Dtype>::dtype Calibrator<Ttype, Dtype>::max_data(Tensor4dPtr<Ttype, Dtype> tensor, int tensor_id) {
//dtype Calibrator<Ttype, Dtype>::max_data(Tensor4dPtr<Ttype, Dtype> tensor ) {
    Tensor4d<X86, Dtype> h_tensor;
    h_tensor.reshape(tensor->valid_shape());
    h_tensor.copy_from(*tensor);
#ifdef USE_CUDA
    cudaDeviceSynchronize();
#endif

    dtype max_value = 0.f;
    auto data = h_tensor.data();
    for (int i = 0; i < h_tensor.valid_size(); i++) {
        //max_value  =  std::max(dtype(abs(data[i])), max_value);
        auto x = fabs(data[i]);
        max_value  =  x > max_value ? x : max_value;
    }
    _max_vec[tensor_id] = _max_vec[tensor_id]  > max_value ? _max_vec[tensor_id] : max_value;
    return _max_vec[tensor_id];
}

template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::histgram(Tensor4dPtr<Ttype, Dtype> tensor, int tensor_id) {
    std::vector<int>& hist_vec = _hist_vecs[tensor_id];
    auto max_value = _max_vec[tensor_id];
    Tensor4d<X86, Dtype> h_tensor;
    h_tensor.reshape(tensor->valid_shape());
    h_tensor.copy_from(*tensor);
    auto data = h_tensor.data();
    auto step = max_value / _bin_num;
    for (int i = 0; i < h_tensor.valid_size(); i++) {
        int id = fabs(data[i]) / step;
        id = id < _bin_num ? id : _bin_num - 1;
        hist_vec[id]++;
    }
}

template<typename Ttype, DataType Dtype>
void Calibrator<Ttype, Dtype>::generate_calibrator_table(std::vector<std::string>& tensor_name_list) {
    int tensor_id = 0;
    for (auto hist : _hist_vecs) {
        dtype min_kl_divergence = 1e30;
        int total_num = 0;
        for(auto a : hist) {
            total_num += a;
        }
        total_num -= hist[0];

        int start_num = 0;
        for (int i = 1; i < 129; i++) {
            start_num += hist[i];
        }

        int thresh = 0;
        dtype kl_array[_bin_num];
        std::vector<float> q;
        std::vector<float> ref_q;
        std::vector<int> ref_p;
        q.reserve(_bin_num);
        ref_q.reserve(_bin_num);
        ref_p.reserve(_bin_num);
        for (int i = 129; i < _bin_num - 1; i++) {
            ref_p.resize(i);
            for(int j = 0; j < ref_p.size(); j++) {
                ref_p[j] = hist[j+1];
            }
            int outlier = total_num - start_num;
            ref_p[i - 1] += outlier;
            int q_size = ref_p.size();
            ref_q.resize(128);
            for(int j = 0; j < ref_q.size(); j++) {
                ref_q[j] = 0.f;
            }
            q.resize(q_size);
            for(int j = 0; j < q_size; j++) {
                q[j] = 0.f;
            }
            get_ref_q(ref_p, ref_q);
            expand_to_q(ref_p, ref_q, q);
            dtype kl = get_kl_divergence(hist, q);
            kl_array[i] = kl;
            thresh = min_kl_divergence > kl ? i : thresh;
            min_kl_divergence  = min_kl_divergence > kl ? kl : min_kl_divergence;
            start_num += hist[i];
        }
        _scale_map.insert(std::pair<std::string, dtype>(tensor_name_list[tensor_id], _max_vec[tensor_id] /_bin_num *  thresh));
        //_scale_map.insert(std::pair<std::string, dtype>(tensor_name_list[tensor_id], thresh));
        tensor_id++;
    }
    write_calibrator();
}

#ifdef USE_CUDA
template class Calibrator<NV, AK_FLOAT>;
#endif
#ifdef USE_X86_PLACE
template class Calibrator<X86, AK_FLOAT>;
#endif
    
}
    
