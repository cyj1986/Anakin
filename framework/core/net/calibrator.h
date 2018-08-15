
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

#ifndef ANAKIN_CALIBRATOR_H
#define ANAKIN_CALIBRATOR_H

#include "framework/core/net/batch_stream.h"
#include "framework/core/base.h"
#include "framework/core/operator/operator.h"
#include <map>

namespace anakin {

/** 
 *  \brief Net class used for execution of graph and it is thread safety.
 */
template<typename Ttype, DataType Dtype>
class Calibrator {
public:
    typedef typename DataTrait<Ttype, Dtype>::dtype  dtype;
    Calibrator(BatchStream<X86, Dtype>* stream,
              int batch_size, int bin_num, std::string calibrator_file){ 
         _batch_stream = stream;
         _batch_size = batch_size;
         _bin_num = bin_num;
         _calibrator_table_file = calibrator_file;
    }
    ~Calibrator() { 
        for (auto tensor : _in_vec) {
             delete tensor;
             tensor = nullptr;
        }
    }
    void init_statistics(int tensor_num);
    int get_batch_data(std::vector<Tensor4dPtr<Ttype, Dtype>> inputs);
    void reset_data_stream();
    int get_batch_size() {return _batch_size;}
    int get_bin_num() {return _bin_num;}
    std::vector<dtype>& max_vec() {return _max_vec;}
    std::vector<std::vector<int>>& hist_vecs() {return _hist_vecs;}

    void read_calibrator();
    void write_calibrator();

    dtype max_data(Tensor4dPtr<Ttype, Dtype> tensor, int tensor_id);

    void histgram(Tensor4dPtr<Ttype, Dtype> tensor, int tensor_id);
    void generate_calibrator_table(std::vector<std::string>& tensor_name_list);
private:
    void get_ref_q(std::vector<int>& ref_p, std::vector<float>& ref_q);
    void expand_to_q(std::vector<int>& ref_p, std::vector<float>& ref_q, std::vector<float>& q);
    dtype get_kl_divergence(std::vector<int>&ref_p, std::vector<float>& q);
    
private:
    BatchStream<X86, Dtype>* _batch_stream;
    std::vector<Tensor4dPtr<X86, Dtype>> _in_vec;
    int _batch_size;
    std::map<std::string, dtype> _scale_map;
    std::string _calibrator_table_file;
    std::vector<dtype> _max_vec;
    std::vector<std::vector<int>> _hist_vecs;
    int _bin_num;
};
}
#endif
