#include <string>
#include "net_test.h"
#include "saber/funcs/timer.h"
#include <chrono>
#include <fstream>
#include <map>

#ifdef USE_CUDA
using Target = NV;
#endif
#ifdef USE_X86_PLACE
using Target = X86;
#endif
#ifdef USE_ARM_PLACE
using Target = ARM;
#endif

// resnet 50
std::string model_path = "/home/cuichaowen/anakin2/anakin2/benchmark/CNN/models/Resnet50.anakin.bin";

std::string src_data_path = "";
std::string trg_data_path = "";
std::string src_dict_path = "";
std::string trg_dict_path = "";
int batch_size = 10;
int n_head = 8;
int beam_size = 5;
int max_length = 30;
int n_best = 1;

std::vector<std::string> string_split(std::string in_str, std::string delimiter) {
    std::vector<std::string> seq;
    int found = in_str.find(delimiter);
    int pre_found = 0;
    while (found != std::string::npos) {
        if (pre_found == 0) {
            seq.push_back(in_str.substr(pre_found, found - pre_found));
        } else {
            seq.push_back(in_str.substr(pre_found + delimiter.length(), found - delimiter.length() - pre_found));
        }
        pre_found = found;
        found = in_str.find(delimiter, pre_found + delimiter.length());
    }
    seq.push_back(in_str.substr(pre_found+1, in_str.length() - (pre_found+1)));
    return seq;
}


class Data{
  public:
      Data(std::vector<std::string> file_list, std::string dict, int batch_size) :
            _file_list(file_list),
            _batch_size(batch_size),
            _file_id(0),
            _pos(0),
            _total_length(0) {
          std::ifstream file(dict);
          if (file.is_open()) {
              std::string str;
              char buf[200];
              while(file.getline(buf, 200)) {
                  str = buf;
                  std::vector<std::string> vec = string_split(str, " ");
                  _dict.insert(std::pair<std::string,int>(vec[0], atoi(vec[1].c_str())));
              }
          }
      } 
      std::vector<std::vector<float>>& get_batch_data();
  private:
      std::vector<std::string> _file_list;
      std::map<std::string, int> _dict;
      int _file_id;
      int _pos;
      int _batch_size;
      int _total_length;
};

std::vector<std::vector<float>>& Data::get_batch_data() {
    std::fstream file(_file_list[_file_id]);
    CHECK(file.is_open()) << "file open failed";
    int seq_num = 0;
    std::vector<std::vector<std::string>> batch_seq;
    
    while (seq_num < _batch_size) {
        std::string s;
        if (_pos == (_total_length -1)) {
            file.close();
            file.open(_file_list[++_file_id]);
            file.seekg(0, file.end);
            _total_length = file.tellg();
            file.seekg(0, file.beg);
            _pos = 0;
           
        } else {
            file.seekg(0, _pos);
        }
        char buf[200];
        while(file.getline(buf, 200)) {
            std::string s = buf;
            if (s.find("seg")) {
                std::vector<std::string> seq;
                int start_pos = s.find_first_of(">");
                int end_pos = s.find_last_of("<");
                if (start_pos != std::string::npos && end_pos != std::string::npos) {
                    std::string real_s = s.substr(start_pos, end_pos - start_pos);
                    seq = string_split(real_s, " ");
                }
                batch_seq.push_back(seq);
                seq_num++;
                if (batch_seq.size() >= _batch_size) {
                    _pos = file.tellg();
                    break;
                }
            }
        }
    }
    std::vector<std::vector<float>> batch;
    for (auto seq :batch_seq) {
        std::vector<float> word_id_vec;
        for (auto word : seq ) {
            word_id_vec.push_back(float(_dict[word]));
        }
        batch.push_back(word_id_vec);
    }
    return batch;
}

void prepare_src_data(std::vector<std::vector<float>>& batch,
           int padding_idx, 
           int n_head, 
           int d_model,
           Tensor<X86, AK_FLOAT, NCHW>* src_word,
           Tensor<X86, AK_FLOAT, NCHW>* src_pos,
           Tensor<X86, AK_FLOAT, NCHW>* src_slf_attn_bias,
           Shape& src_data_shape) {
    std::vector<int> offset;
    offset.push_back(0);
    int num_words = 0;
    int max_seq_len = 0;
    for (auto seq : batch) {
        num_words += seq.size();
        offset.push_back(num_words);
        max_seq_len = std::max(size_t(seq.size()), size_t(max_seq_len));
    }
    int num_words_total = batch.size() * max_seq_len;
    Shape shape = {num_words_total, 1, 1, 1};
    src_word->reshape(shape);
    src_pos->reshape(shape);
    Shape src_slf_attn_bias_shape = {batch.size(), n_head, max_seq_len, max_seq_len};
    src_slf_attn_bias->reshape(src_slf_attn_bias_shape);
    auto word_data = src_word->mutable_data();
    auto pos_data = src_pos->mutable_data();
    auto bias_data = src_slf_attn_bias->mutable_data();
    int count = 0;
    for (int i = 0; i < batch.size(); i++) {
        for (int j = 0; j < batch[i].size(); j++) {
            word_data[count] = batch[i][j];
            pos_data[count] = j;
            count++;
        }
        for (int j = batch[i].size(); j < max_seq_len; j++) {
            word_data[count] = padding_idx;
            pos_data[count] = 0;
            count++;
        }
    }
    count = 0;
    for (int n = 0; n < batch.size(); n++) {
        for (int c = 0; c < n_head; c++) {
            for (int h = 0; h < max_seq_len; h++) {
                for (int w = 0; w < batch[n].size(); w++) {
                    bias_data[count++] = 0;      
                }
                for (int w = batch[n].size(); w < max_seq_len; w++) {
                    bias_data[count++] = -1e9;      
                }
            }
        }
    }
    src_data_shape = {batch.size(), max_seq_len, 1, d_model};
}

void  init_dec_in_data(int batch_size, int beam_size, int bos_idx, int d_model, int num_head,
            Tensor<X86, AK_FLOAT, NCHW>* src_word,
            Tensor<X86, AK_FLOAT, NCHW>* src_pos,
            Tensor<X86, AK_FLOAT, NCHW>* src_slf_attn_bias,
            Tensor<X86, AK_FLOAT, NCHW>* enc_out, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_words, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_pos, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_slf_attn_bias, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_src_attn_bias,
            Shape& trg_data_shape
            ) {
    int src_max_len = src_slf_attn_bias->width();
    int trg_max_len = 1;
    Shape trg_words_shape = {batch_size * beam_size, 1, 1, 1};
    trg_words->reshape(trg_words_shape); 
    trg_pos->reshape(trg_words_shape); 
    auto trg_word_data = trg_words->mutable_data();
    auto trg_pos_data = trg_pos->mutable_data();
    for (int i = 0; i < trg_words_shape.count(); i++) {
        trg_word_data[i] = bos_idx;
        trg_pos_data[i] = 1;
    }
    Shape trg_slf_attn_bias_shape = {batch_size* beam_size, num_head, trg_max_len, trg_max_len};
    trg_slf_attn_bias->reshape(trg_slf_attn_bias_shape);
    auto trg_slf_attn_bias_data = trg_slf_attn_bias->mutable_data();
    for (int i = 0; i < trg_slf_attn_bias->valid_size(); i++) {
        trg_slf_attn_bias_data[i] = 0;
    }
    auto trg_src_attn_bias_data = trg_src_attn_bias->mutable_data();
    auto src_slf_attn_bias_data = src_slf_attn_bias->data();
    int count = 0;
    for (int n = 0; n < batch_size; n++) {
        for (int b = 0; b < beam_size; b++) {
            for (int c = 0; c < num_head; c++) {
                int src_offset = (n * num_head + c )* src_max_len * src_max_len;
                for (int t = 0; t < trg_max_len; t++) {
                    for (int s = 0; s < src_max_len; s++) {
                       trg_src_attn_bias_data[count++] = src_slf_attn_bias_data[src_offset + s];
                         
                    }
                }
            }
        }
    }
    trg_data_shape = {batch_size, beam_size, trg_max_len, d_model};
}

std::vector<std::vector<int>>  beam_back_trace(std::vector<std::vector<int>>& prev_branchs,
                     std::vector<std::vector<int>>& next_ids,
                     const int bos_idx, 
                     const int n_best) {
    std::vector<std::vector<int>>seqs;
    for (int i = 0; i < n_best; i++) {
        int k = i;
        std::vector<int> seq;
        seq.resize(prev_branchs.size() + 1);
        seq[0] = bos_idx;
        for (int j = prev_branchs.size() - 1; j >= 0; j--) {
            seq[j+1] = next_ids[j][k];
            k = prev_branchs[j][k];
        }
        seqs.push_back(seq);
    }
    return seqs;
}

void update_decode_in_data (
            Tensor<X86, AK_FLOAT, NCHW>* trg_words, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_pos, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_slf_attn_bias, 
            Tensor<X86, AK_FLOAT, NCHW>* trg_src_attn_bias,
            Tensor<X86, AK_FLOAT, NCHW>* enc_out,
            Shape& trg_data_shape,
            std::vector<std::vector<std::vector<int>>>& next_ids,
            std::vector<std::vector<std::vector<int>>>& prev_branchs,
            std::vector<int>& active_beams,
            std::map<int, int>& beam_inst_map,
            int beam_size,
            int bos_idx
            ) {
    int trg_cur_len = trg_slf_attn_bias->width() + 1;
    Shape trg_shape = {active_beams.size() * beam_size * trg_cur_len, 1, 1, 1};
    trg_words->reshape(trg_shape);
    trg_pos->reshape(trg_shape);
    auto word = trg_words->mutable_data();
    auto pos = trg_pos->mutable_data();
    for (int i = 0; i < active_beams.size(); i++) {
        auto vec = beam_back_trace(prev_branchs[active_beams[i]],
                next_ids[active_beams[i]],
                bos_idx,
                n_best);
        for (int j = 0; j < beam_size; j++) {
            for (int k = 0; k < trg_cur_len + 1; k++) {
                *word++ = vec[j][k];
                *pos++ = k;
            }
        }
    }
    Shape trg_slf_attn_bias_shape = {active_beams.size() * beam_size, n_head, trg_cur_len, trg_cur_len};
    trg_slf_attn_bias->reshape(trg_slf_attn_bias_shape);
    auto trg_bias_data = trg_slf_attn_bias->mutable_data();
    for (int i = 0; i < active_beams.size() * beam_size * n_head; i++) {
        for (int h = 0; h < trg_cur_len; h++) {
            for (int w = 0; w < h+1; w++) {
                *trg_bias_data++ = 0;
            }
            for (int w = h+1; w < trg_cur_len; w++) {
                *trg_bias_data++ = -1e9;
            }
        }
    } 
    Shape trg_src_attn_bias_shape = trg_src_attn_bias->valid_shape();
    trg_src_attn_bias_shape[0] = active_beams.size() * beam_size;
    trg_src_attn_bias_shape[2] = trg_cur_len;
    auto trg_src_data = trg_src_attn_bias->data();
    Tensor<X86, AK_FLOAT, NCHW> new_trg_src_attn_bias;
    int old_trg_len = trg_src_attn_bias->valid_shape()[2];
    new_trg_src_attn_bias.reshape(trg_src_attn_bias_shape);
    auto new_trg_src_data = new_trg_src_attn_bias.data();
    for (int i = 0; i < active_beams.size(); i++) {
        int inst_id = beam_inst_map[active_beams[i]];
        for (int j = 0; j < beam_size; j++) {
            int id = inst_id * beam_size + j;
            for (int k = 0; k < trg_src_attn_bias_shape[1]; k++) {
                int new_start = ((i * beam_size + j) * trg_src_attn_bias_shape[1] + k ) * trg_src_attn_bias_shape[2];
                int old_start = (id * trg_src_attn_bias_shape[1] + k) * trg_src_attn_bias_shape[2];
                for (int m = 0; m < trg_src_attn_bias_shape[2]; m++) {
                    memcpy(new_trg_src_data + new_start * trg_src_attn_bias_shape[3], trg_src_data + (old_start + (old_trg_len - 1)) * trg_src_attn_bias_shape[3], sizeof(float) * trg_src_attn_bias_shape[3]);
                    new_start++;
                }
            }
        }
    }
    trg_src_attn_bias->reshape(trg_src_attn_bias_shape);
    trg_src_attn_bias->copy_from(new_trg_src_attn_bias);
    Tensor<X86, AK_FLOAT, NCHW> new_enc_out;
    Shape new_enc_out_shape = enc_out->valid_shape();
    new_enc_out_shape[0] = active_beams.size() * beam_size;
    new_enc_out.reshape(new_enc_out_shape);
    auto new_enc_out_data = new_enc_out.mutable_data();
    auto old_enc_out_data = enc_out->data();
    int remain_count  = new_enc_out.valid_size() / new_enc_out.num();
    for (int i  = 0 ; i < active_beams.size(); i++) {
        for (int j = 0; j < beam_size; j++) {
            memcpy(new_enc_out_data + (i * beam_size + j) * remain_count, old_enc_out_data + (beam_inst_map[i] * beam_size + j) * remain_count, sizeof(float) * remain_count);
        }
    }
    enc_out->reshape(new_enc_out_shape);
    enc_out->copy_from(new_enc_out);
    
}

struct Data_ID {
float data;
int id;
};
bool compare(Data_ID d1, Data_ID d2) {
    return d1.data > d2.data;
}

void beam_search(Tensor<X86, AK_FLOAT, NCHW>* predict, 
                int cur_seq_len, std::vector<std::vector<int>> scores, int beam_size, 
                bool output_unk,
                int unk_ids,
                int eos_ids,
                std::map<int, int> beam_inst_map, 
                std::vector<int>& active_beams,
                std::vector<std::vector<std::vector<int>>>& prev_branches,
                std::vector<std::vector<std::vector<int>>>& next_ids) {
    int seq_len = cur_seq_len + 1;
    Shape shape = {beam_inst_map.size() , beam_size, seq_len, predict->valid_size() /(beam_inst_map.size() * beam_size *(seq_len))};
    predict->reshape(shape);
    Tensor<X86, AK_FLOAT, NCHW> predict_all;
    Shape last_shape = shape;
    last_shape[2] = 1;
    predict_all.reshape(last_shape);
    active_beams.clear();
    auto data_in = predict->data();
    auto data_out = predict_all.mutable_data();
    for (int i = 0; i < beam_inst_map.size() * beam_size; i++) {
        auto tmp_data_out = data_out + i * shape[3];
        auto tmp_data_in = data_out + ((i + 1) * seq_len - 1) * shape[3];
        for ( int j = 0; j < shape[3]; j++) {
            tmp_data_out[j] = log(tmp_data_in[j]);
       }
       if (!output_unk) {
           tmp_data_out[unk_ids] = -1e9;
       }
    }
    // score;
    int count = 0;
   for (int i = 0; i < active_beams.size(); i++) {
        for (int beam_id = 0; beam_id  < beam_size; beam_id++) {
            float score = scores[active_beams[i]][beam_id];
            for (int k = 0; k < shape[3]; k++) {
                 data_out[count++] += score;
            }
        }
    }
    for (int i = 0; i < batch_size; i++) {
        if (beam_inst_map.find(i) == beam_inst_map.end()) {
             continue;
        }
        auto inst_idx = beam_inst_map[i];
        auto seq_proc = data_out + inst_idx * beam_size * shape[3];
        std::vector<Data_ID> vec;
        for (int k = 0; k < beam_size * shape[3]; k++) {
              Data_ID data_id;
              data_id.data = seq_proc[k];
              data_id.id = k;
              vec.push_back(data_id);
        }
        std::partial_sort(vec.begin(), vec.begin() + beam_size, vec.end(), compare);
        std::vector<int> pre_branch_vec;
        std::vector<int> next_ids_vec;
        std::vector<int> cur_seq_scores;
        for (int beam = 0; beam < beam_size; beam++) {
            cur_seq_scores.push_back(vec[0].data);
            next_ids_vec.push_back(vec[0].id % shape[3]);
            pre_branch_vec.push_back(vec[0].id / shape[3]);
        }
        next_ids[i].push_back(next_ids_vec);
        prev_branches[i].push_back(pre_branch_vec);
        scores[i] = cur_seq_scores;
        if (next_ids_vec[0] != eos_ids ) {
            active_beams.push_back(i);
        }
    }
    beam_inst_map.clear();
    for (int i = 0; i < active_beams.size(); i++) {
        beam_inst_map.insert(std::pair<int, int>(active_beams[i], i));
    }
}








TEST(NetTest, net_execute_base_test) {
    graph = new Graph<Target, AK_FLOAT, Precision::FP32>();
    LOG(WARNING) << "load anakin model file from " << model_path << " ...";
    // load anakin model files.
    auto status = graph->load(model_path);
    if(!status ) {
        LOG(FATAL) << " [ERROR] " << status.info();
    }

    // reshape the input_0 's shape for graph model
    //graph->Reshape("input_0", {1, 8, 640, 640});

    // register all tensor inside graph
    //graph->RegistAllOut();

    //anakin graph optimization
    graph->Optimize();

    // constructs the executer net
    Net<Target, AK_FLOAT, Precision::FP32> net_executer(*graph, true);

    // get in
    auto d_tensor_in_p = net_executer.get_in("input_0");
    Tensor4d<X86, AK_FLOAT> h_tensor_in;

    auto valid_shape_in = d_tensor_in_p->valid_shape();
    for (int i=0; i<valid_shape_in.size(); i++) {
        LOG(INFO) << "detect input_0 dims[" << i << "]" << valid_shape_in[i];
    }

    h_tensor_in.re_alloc(valid_shape_in);
    float* h_data = h_tensor_in.mutable_data();

    for (int i=0; i<h_tensor_in.size(); i++) {
        h_data[i] = 1.0f;
    }

    d_tensor_in_p->copy_from(h_tensor_in);


    int epoch = 1;
    // do inference
    Context<Target> ctx(0, 0, 0);
    LOG(WARNING) << "EXECUTER !!!!!!!! ";

    net_executer.prediction();


	LOG(ERROR) << "inner net exe over !";

    //auto& tensor_out_inner_p = net_executer.get_tensor_from_edge("data_perm", "conv1");

    // get out yolo_v2
    /*auto tensor_out_0_p = net_executer.get_out("loc_pred_out");
    auto tensor_out_1_p = net_executer.get_out("obj_pred_out");
    auto tensor_out_2_p = net_executer.get_out("cls_pred_out");
    auto tensor_out_3_p = net_executer.get_out("ori_pred_out");
    auto tensor_out_4_p = net_executer.get_out("dim_pred_out");*/

	// get outs cnn_seg 
	/*auto tensor_out_0_p = net_executer.get_out("slice_[dump, mask]_out");
	auto tensor_out_1_p = net_executer.get_out("category_score_out");
	auto tensor_out_2_p = net_executer.get_out("instance_pt_out");
   	auto tensor_out_3_p = net_executer.get_out("confidence_score_out");
	auto tensor_out_4_p = net_executer.get_out("class_score_out");
	auto tensor_out_5_p = net_executer.get_out("heading_pt_out");
	auto tensor_out_6_p = net_executer.get_out("height_pt_out");*/
    // get out result
    //test_print(tensor_out_4_p);


    // save the optimized model to disk.
    /*std::string save_model_path = model_path + std::string(".saved");
    status = graph->save(save_model_path);
    if (!status ) { 
        LOG(FATAL) << " [ERROR] " << status.info(); 
    }*/
}


int main(int argc, const char** argv){
    // initial logger
    logger::init(argv[0]);
	InitTest();
	RUN_ALL_TESTS(argv[0]);	
	return 0;
}
