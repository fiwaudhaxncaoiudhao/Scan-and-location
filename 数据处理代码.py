import pandas as pd

# 读取536测试数据
df_raw = pd.read_excel('536测试.xlsx', sheet_name='Sheet1', header=None)

records = []
current_label = None
current_location = []
current_rssi_dict = {}
in_data = False
mac_columns = set()

# 逐行解析数据
for i, row in df_raw.iterrows():
    line = [str(x).strip() for x in row.values if str(x).strip() != '']
    if not line:
        continue
    
    # 遇到WiFi扫描分隔符，保存上一组数据
    if line[0].startswith('===== WiFi Scan Results'):
        if current_label is not None and current_rssi_dict:
            record = {
                'Label': current_label,
                'X': current_location[0],
                'Y': current_location[1],
                'Z': current_location[2]
            }
            # 同一MAC多次扫描取平均值
            for mac, rssi_list in current_rssi_dict.items():
                record[mac] = sum(rssi_list) / len(rssi_list)
            records.append(record)
            mac_columns.update(current_rssi_dict.keys())
        # 重置
        current_rssi_dict = {}
        in_data = False
        continue
    
    # 提取Label
    if line[0].startswith('Label'):
        current_label = int(line[1])
        continue
    
    # 提取坐标Location
    if line[0].startswith('Location'):
        current_location = [float(line[1]), float(line[2]), float(line[3])]
        continue
    
    # 标记进入RSSI数据区
    if line[0].startswith('SSID'):
        in_data = True
        continue
    
    # 提取JXUST-WLAN的MAC与RSSI
    if in_data and len(line) >= 4 and line[0] == 'JXUST-WLAN':
        mac = line[2]
        rssi = int(line[1])
        if mac not in current_rssi_dict:
            current_rssi_dict[mac] = []
        current_rssi_dict[mac].append(rssi)

# 处理最后一组数据
if current_label is not None and current_rssi_dict:
    record = {
        'Label': current_label,
        'X': current_location[0],
        'Y': current_location[1],
        'Z': current_location[2]
    }
    for mac, rssi_list in current_rssi_dict.items():
        record[mac] = sum(rssi_list) / len(rssi_list)
    records.append(record)
    mac_columns.update(current_rssi_dict.keys())

# 生成宽表，缺失MAC填充-100
df_wide = pd.DataFrame(records)
sorted_macs = sorted(mac_columns)

for mac in sorted_macs:
    if mac not in df_wide.columns:
        df_wide[mac] = -100.0

# 固定列顺序
df_wide = df_wide[['Label', 'X', 'Y', 'Z'] + sorted_macs]
df_wide = df_wide.fillna(-100.0)

# 保存结果
df_wide.to_excel('536测试_宽表结果.xlsx', index=False)
print("536测试.xlsx 处理完成 → 已保存：536测试_宽表结果.xlsx")