import numpy as np 
import cv2 
import pandas as pd
import random
from cnn.ConvNet import *
from cnn.common.optimizer import RMSProp
from module.train import *
from module.segment import *
from module.builder import *
import pickle
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image
import glob
from fl_detector      import FLDetector
from shapley_detector import compute_shap_values, filter_by_shap
from gnn_detector     import detect as gnn_detect, build_client_graph
from ledger           import add_block
import logging
import torch.nn.functional as F
import torch
import tenseal as tsl
import json
import time
import io

def measure_downlink(server: str) -> int:
     """返回最新 global.npy 文件大小（字节）"""
     path = f"{server}/global.npy"
     return os.path.getsize(path)

def measure_uplink(enc) -> int:
     """返回 CKKS 向量序列化后字节长度"""
     buf = enc.serialize()
     return len(buf)

context = tsl.context(
    tsl.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = 2**40
context.generate_galois_keys()

logging.basicConfig(
    filename="malicious.log",
    level=logging.INFO,
    format="%(asctime)s [Round:%(round)d][Server:%(server)s] Malicious: %(clients)s"
)
def log_malicious(round: int, server: str, clients: list):
    logging.info("", extra={
        "round":   round,
        "server":  server,
        "clients": ",".join(clients)
    })
# Retrieve the global model
def download(server):
    global_params = np.load("%s/global.npy" %server, allow_pickle = True).item()

    return global_params 
    

# Retrieve the local model
def upload(node_name):  
    local_params = np.load("%s/local.npy" %node_name, allow_pickle = True).item() 
    
    return local_params


# Evaluation
def precision(local_params, train_data, label):
    label = np.array(label)
    
    snet = ConvNet(input_dim=(1, 48, 48), 
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = False, dropout_ration = 0.1, use_batchnorm = False)
  
    snet.params = local_params
    snet.layer()
    acc = snet.accuracy(train_data, label, 125)
    
    return acc
    

# Update local models
def Execute(node_name, pcap, server, pattern):
    local_params_init = download(server)
    train(node_name, pcap, local_params_init, pattern)
    local_params = upload(node_name)  # dict of np.ndarray

    # 2) 将 dict 展平为 1D list
    flat = []
    for w in local_params.values():
        flat.extend(w.ravel().tolist())

    # 3) CKKS 加密
    enc = tsl.ckks_vector(context, flat)
    return enc


# Segmented Model Aggregation
def Aggregate(local_parameters_list, global_parameters, server_list, server):
    """
    Federated Learning with Adaptive Weight Aggregation (FedAWA)，
    根据每个客户端更新与全局方向的一致性动态分配聚合权重。 :contentReference[oaicite:1]{index=1}
    """
    # 1) 计算每个客户端更新 delta = local - global
    deltas = []
    for lp in local_parameters_list:
        deltas.append({k: lp[k] - global_parameters[k]
                       for k in global_parameters})

    # 2) 将每个 delta 拼接成向量并计算与全局方向的余弦相似度
    scores = []
    for delta in deltas:
        vec = torch.cat([torch.from_numpy(delta[k].ravel()).float()
                         for k in delta], dim=0)
        # 以全零向量模拟“全局方向”
        glob = torch.zeros_like(vec)
        score = F.cosine_similarity(vec, glob, dim=0).item()
        scores.append(score)

    # 3) 对齐度转为权重（softmax 归一化）
    weights = F.softmax(torch.tensor(scores), dim=0).numpy()

    # 4) 加权聚合
    new_global = {}
    for k in global_parameters:
        new_global[k] = sum(weights[i] * local_parameters_list[i][k]
                            for i in range(len(local_parameters_list)))

    # 5) 保存新的全局模型
    np.save(f"{server}/global", new_global)

# Retrieve image paths based on the node name and the anomaly type 
def readList(node, pattern):
    dirpath ="%s" %node
    li_fpath = sorted(glob.glob(os.path.join(dirpath, "pcap", "*_%s" %pattern)))
    
    return li_fpath






def main():
    # The dataset we used for training
    global_acc_list = []
    remove_rate_list = []  # 每轮剔除客户端比例
    round_time_list  = []  # 每轮耗时（秒）
    node_list = ["n005", "n008",  "n036", "n047", "n006",  "n034", "n038", "n045", "n048",  "n031", "n035", "n041", "n046", "n053", "n056"]
    
    traffic_list     = []  # 每轮通信量（字节）
    train_delay_list = []  # 本地训练延迟（秒）
    agg_delay_list   = []  # 聚合延迟（秒） 
    plain_traffic_list  = []  # 每轮明文上传（Bytes）
    cipher_traffic_list = []  # 每轮密文上传（Bytes）



    Knowledge_type = input("Enter an anomaly type for training (select from  TypeA,  TypeB): ")
    Knowledge_types = {"TypeA": 1, "TypeB": 3}
    pattern = Knowledge_types[Knowledge_type]


    # We define hyperparameters and initialize the global model
    max_rounds = 20
    global_params = {}
    print("Systems initialize...")
    global_params = np.load("init/init.npy", allow_pickle = True).item()
    
    osNet =ConvNet(input_dim=(1, 48, 48),
                 conv_param={'filter1_num':10, 'filter1_size':3, 'filter2_num':10, 'filter2_size':1, 'pad':1, 'stride':1},
                 hidden_size= 200, output_size= 2 ,use_dropout = True, dropout_ration = 0.1, use_batchnorm = False)

    osNet.params = global_params
    osNet.layer()
    
    model_name = '000/global'
    np.save(model_name, osNet.params) 
    
    for i in range(len(node_list)):
        model_name = "%s/local.npy" %node_list[i]
        np.save(model_name, osNet.params) 
        
    detector = FLDetector(threshold=0.05)
    anomaly_scores_dict = {}   # { server_name: {client_id: score, …}, … }

    # The server list contains the segmentation info.
    server_list = {"000": node_list}

    with open("server_dict.txt", "wb") as myFile:
        pickle.dump(server_list, myFile)
    
    acc_dict = {}
    for i in range(len(node_list)):
        acc_dict[node_list[i]] = []
    full_acc_dict = {}
    for i in range(len(node_list)):
        full_acc_dict[node_list[i]] = []
    acc_list = []



    # Start training
    for i in range(max_rounds):
        start_time = time.time()
        sum_up_bytes = 0
        sum_plain_bytes  = 0
        sum_cipher_bytes = 0
        train_sum = 0.0
        train_count = 0
        agg_sum = 0.0
        agg_count = 0

        # Read current segmentation info. from the txt file. 
        with open("server_dict.txt", "rb") as myFile:
            server_list = pickle.load(myFile)
        
        # For each existing server, we perform FL.    
        for s in range(len(getListKeys(server_list))):
            server_name = getListKeys(server_list)[s] # Server name
            node_num = len(getListValues(server_list)[s]) # Node numbers
            node_list = getListValues(server_list)[s] # Nodes assigned to the server
       


            # Randomly select a group of train_num clients for updating every round 
            train_num = (node_num+1)//2
            select_client = [] 
            clients = []
            remaining_list = node_list
            
            contrib = {cid: sum(acc_dict[cid]) for cid in node_list}
            sorted_clients = sorted(node_list,
                                    key=lambda c: contrib[c],
                                    reverse=True)
            # 这里存的是原 node_list 中元素的索引
            select_client = [ node_list.index(c) for c in sorted_clients[:train_num] ]
    
            for j in range(train_num):
                with open("server_dict.txt", "rb") as myFile:
                     server_list = pickle.load(myFile)
                node_list = getListValues(server_list)[s]
                clients.append(node_list[select_client[j]])
                remaining_list.remove(node_list[select_client[j]])
            
            print("[Round:%s][Server:%s] Clients %s will update local models..." %(i+1, server_name, clients))
            
            
            # Retrieve the latest global model
            global_parameters = download(server_name) 
            down_bytes = measure_downlink(server_name)

            # Broadcast the global model to clients for evaluation  
            # Selected clients every round
            for j in range(len(clients)):

                # Retrieve the client local data
                data_dir = readList(clients[j], pattern)[i]
                data, label = makedataset("%s/dataset" %(data_dir))
                # Local model
                local_params = upload(clients[j])
                
                # Evaluation on the client model with acc
                acc = precision(local_params, data, label)

                acc_dict[clients[j]].append(acc)
                full_acc_dict[clients[j]].append(acc)
                
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, clients[j], acc))
                
                
            # The remaining clients
            for j in range(len(remaining_list)):

                # Retrieve the latest global model
                model_name = "%s/local.npy" %remaining_list[j]
                np.save(model_name, global_parameters) 
                
                # Retrieve the local client data 
                data_dir = readList(remaining_list[j], pattern)[i]
                data, label = makedataset("%s/dataset" %(data_dir))
                # Local model
                local_params = upload(remaining_list[j])
                
                # Evaluation on the client model with acc
                acc = precision(local_params, data, label)

                acc_dict[remaining_list[j]].append(acc)
                full_acc_dict[remaining_list[j]].append(acc)
                
                print("[Round:%s][Server:%s][%s] ML detection precision: %s" %(i+1, server_name, remaining_list[j], acc))
               
            
            # Perform local model training         
            local_parameters_list = []
            for j, cid in enumerate(clients):
                print(f"[Round:{i+1}][Server:{server_name}][{cid}] Training and encrypting...")

                params = upload(cid)
                buf = io.BytesIO()
                np.save(buf, params)
                sum_plain_bytes += buf.tell()
           # 1) 执行加密并计时
                t0 = time.perf_counter()
                enc = Execute(cid, readList(cid, pattern)[i].replace(f"_{pattern}", ""), server_name, pattern)
                t1 = time.perf_counter()
                train_sum += (t1 - t0)
                train_count += 1
                local_parameters_list.append(enc)

            # 2) 累计上行字节
                sum_up_bytes += measure_uplink(enc)
                cipher_buf = enc.serialize()
                sum_cipher_bytes += len(cipher_buf)
 

            decrypted_updates = []
            for enc_upd in local_parameters_list:
                flat = enc_upd.decrypt()
                upd = {}
                offset = 0
                for k, g in global_parameters.items():
                    size = g.size
                    arr = np.array(flat[offset:offset+size]).reshape(g.shape)
                    upd[k] = arr
                    offset += size
                decrypted_updates.append(upd)

            # 逐对处理：保证不越界
            t0 = time.perf_counter()
            for cid, upd in zip(clients, decrypted_updates):
                detector.record(cid, upd)
            bad1 = detector.detect()

    # 2) Shapley 值贡献检测
            shap_vals = compute_shap_values(
    global_parameters,
    dict(zip(clients, decrypted_updates)))
            bad2 = filter_by_shap(shap_vals)

    # 3) GNN 异常节点检测
            client_feats, edge_index = build_client_graph(decrypted_updates, clients)
            bad3_idx = gnn_detect(client_feats, edge_index)          # [0,2,5,...]
            bad3     = [clients[i] for i in bad3_idx]
            anomaly_scores = {cid: abs(shap_vals.get(cid, 0)) for cid in clients}
            for cid in set(bad1 + bad2 + bad3):
                anomaly_scores[cid] = anomaly_scores.get(cid, 0) + 1.0
            anomaly_scores_dict[server_name] = anomaly_scores

    # 4) 记录区块链审计
            for idx, cid in enumerate(clients):
                add_block(cid, local_parameters_list[idx])

    # 汇总所有可疑客户端，剔除其模型更新
            bad = sorted(set(bad1) | set(bad2) | set(bad3))
            safe_updates = []
            safe_clients = []
            for idx, cid in enumerate(clients):
                if cid not in bad:
                   safe_clients.append(cid)
                   safe_updates.append(local_parameters_list[idx])
    # 记录溯源日志
            log_malicious(i+1, server_name, bad)

    # 用剔除后的更新列表继续聚合
            local_parameters_list = safe_updates
            clients = safe_clients    
               
            # Update the list of neighboring servers
            other_server_list = getListKeys(server_list)
            other_server_list.pop(s)
            
            
            if not decrypted_updates:
        # 没有本地更新，退回使用上一轮全局模型
                print(f"[Round:{i+1}][Server:{server_name}] 无安全更新，用全局模型保持不变")
                np.save(f"{server_name}/global.npy", global_parameters)
            else:
            # Model aggregation to update the global model based on the updated local models
                t_agg0 = time.perf_counter()
                Aggregate(decrypted_updates, global_parameters, other_server_list, server_name)
                t_agg1 = time.perf_counter()
                agg_sum += (t_agg1 - t_agg0)
                agg_count += 1
                # agg_delay_list.append(t_agg1 - t_agg0)

            
            # One round of the current server is completed
            print("[Round:%s][Server:%s] Aggregating successfully" % (i+1, server_name))  

        avg_acc = np.mean([np.mean(acc_dict[cid]) for cid in node_list])
        global_acc_list.append(avg_acc)    
        remove_rate_list.append(len(bad) / max(1, len(clients)))
         # 3) 本轮耗时
        traffic_list.append(down_bytes + sum_up_bytes)
        plain_traffic_list.append(sum_plain_bytes)
        cipher_traffic_list.append(sum_cipher_bytes)
        round_time_list.append(time.time() - start_time)
        if train_count > 0:
            train_delay_list.append(train_sum / train_count)
        else:
            train_delay_list.append(0.0)
        if agg_count > 0:
            agg_delay_list.append(agg_sum / agg_count)
        else:
            agg_delay_list.append(0.0)
        # 清空本轮的 acc_dict
        for c in node_list:
            acc_dict[c].clear()

            
        # For every six rounds, we evaluate the performance of nodes under each server, and depending on their performance update the segmentation list  
        if (i+1)%6 == 0:
            
            fineness = 2
            
            # Compute the average acc of the nodes in each server    
            for k in range(len(getListKeys(acc_dict))):
                acc_dict[getListKeys(acc_dict)[k]] = np.mean(acc_dict[getListKeys(acc_dict)[k]])    
            
            for k in range(len(getListValues(server_list)[s])):
                acc_list.append(acc_dict[getListValues(server_list)[s][k]])
            
            print("[Round:%s][Server:%s] Average accuarcy:%s" %(i+1, server_name, acc_list))

            
            # Depending on the average acc, we set the threshold and update the segmentation list 
            print("[Round:%s] Start rearranging nodes" %(i+1))

            # Find out the nodes with acc scores under the threshold in each server     
            anomaly_scores = anomaly_scores_dict.get(server_name, {})
            index, server = dropout(acc_list, getListKeys(server_list), 0.5-fineness/100, acc_dict,anomaly_scores)
                        # If such nodes exist, update the segmentation list 
            if len(index) > 0:
                for p in range(len(index)):
                    server_update(server_list, getListKeys(acc_dict)[index[p]], server)

            # Reset the acc metric list
            for k in range(len(getListKeys(acc_dict))):
                    acc_dict[getListKeys(acc_dict)[k]] = []             
            acc_list = []  

        print("===============================================")

    pd.DataFrame({
        'round': list(range(1, len(global_acc_list)+1)),
        'avg_accuracy': global_acc_list
    }).to_csv('training_metrics.csv', index=False)

    # 2) 汇总并排序异常得分，取 Top5
    total_scores = {}
    for scores in anomaly_scores_dict.values():
        for cid, sc in scores.items():
            total_scores[cid] = total_scores.get(cid, 0) + sc
    ranked = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    top5 = ranked[:5]

    # 3) 打印并保存最可疑节点
    print("=== 最可疑客户端 Top5 ===")
    with open('suspect_nodes.csv', 'w') as f:
        f.write("client_id,score\n")
        for cid, sc in top5:
            print(f"{cid}: 累积异常分数 = {sc:.3f}")
            f.write(f"{cid},{sc:.3f}\n")
    print("训练指标已保存到 training_metrics.csv；可疑节点已保存到 suspect_nodes.csv")
    
    rounds = list(range(1, len(global_acc_list)+1))
    plt.figure(figsize=(10,6))
    #plt.plot(rounds, global_acc_list,   label='Ave Precision', marker='o')
    plt.plot(rounds, remove_rate_list,   label='Rejection',       marker='x')
    plt.plot(rounds, round_time_list,    label='Overall Time(s)',  marker='s')
    plt.plot(rounds, traffic_list,       label='Traffic (Bytes)',  marker='d')
    plt.plot(rounds, train_delay_list,   label='Avg Train Delay(s)',   marker='o')
    plt.plot(rounds, agg_delay_list,     label='Agg Delay(s)',     marker='^')
    plt.xlabel('Round')
    plt.legend()
    plt.title(' chart')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('core_metrics_curve.png')
    print("已保存：core_metrics_curve.png")
    df = pd.DataFrame({
        'round':            rounds,
        'avg_accuracy':     global_acc_list,
        'remove_rate':      remove_rate_list,
        'overall_time_s':   round_time_list,
        'plain_bytes':      plain_traffic_list,
        'cipher_bytes':     cipher_traffic_list,
        'train_delay_s':    train_delay_list,
        'agg_delay_s':      agg_delay_list,
        'all_bytes':        traffic_list   })
    df.to_csv('all_metrics.csv', index=False)
    print("已保存所有指标到 all_metrics.csv")


    SUSPICION_THRESHOLD = 1.0

# 构建节点列表
    nodes = []
    for sv, scores in anomaly_scores_dict.items():
        nodes.append({"id": sv, "type": "server", "suspicious": False})
        for cid, sc in scores.items():
            nodes.append({
                "id": cid,
                "type": "client",
                "suspicious": bool(sc > SUSPICION_THRESHOLD)
            })
# 去重
    nodes = list({n["id"]: n for n in nodes}.values())

# 构建连边列表
    links = []
    for sv, scores in anomaly_scores_dict.items():
        for cid in scores:
            links.append({"source": cid, "target": sv})

# 写入 JSON
    with open("network_nodes.json", "w") as f:
        json.dump(nodes, f, indent=2)
    with open("network_links.json", "w") as f:
        json.dump(links, f, indent=2)

    print("Wrote network_nodes.json and network_links.json for D3.js visualization")


if __name__ == '__main__':
    main()
