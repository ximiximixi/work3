# 端盖缺陷标注前端

这个小工具只显示原视频右半边裁剪图，标注坐标会自动加上 `offset_x=960`，保存为原始视频坐标。

## 启动

```powershell
cd C:\Users\11816\Desktop\cv\work3\cap_annotation_tool
C:\Users\11816\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe .\server.py
```

打开：

```text
http://127.0.0.1:8765/
```

## 标注建议

- 帧标签：`NORMAL / DEFECT / UNKNOWN`
- 缺陷定义：应该有米白色套/盖的位置没有看到
- 看不清、遮挡、截断、镜像干扰：标 `UNKNOWN`
- 可以画 `sample` 框、`top/bottom/side_upper/side_lower` 框，也可以画 `fixed_region` 固定检测区域
- 点击 `保存` 会写入 `annotations.json`

## 快捷键

- `A/D`：上一帧 / 下一帧
- `1/2/3`：当前帧标为 `NORMAL / DEFECT / UNKNOWN`
- `Delete`：删除选中框
