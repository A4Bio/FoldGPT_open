import os

# 定义PDB文件夹路径和图片保存路径
pdb_folder = "./"  # 替换为你的PDB文件所在文件夹的路径
output_folder = "./"  # 替换为你想保存图片的文件夹路径

# 如果输出文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取文件夹中的所有PDB文件
for pdb_file in os.listdir(pdb_folder):
    if pdb_file.endswith(".pdb"):
        # 加载PDB文件
        cmd.load(os.path.join(pdb_folder, pdb_file))

        # 定义输出图片文件名（保持与PDB文件名一致，但后缀为.png）
        output_file = os.path.join(output_folder, pdb_file.replace(".pdb", ".png"))
        
        # 渲染并保存图片
        cmd.png(output_file)
        
        # 清除当前加载的对象
        cmd.delete("all")