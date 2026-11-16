# bravegpuwinner 团队报告审阅源码

本分支共享到团队自有 fork，供组员整理报告和审阅代码。官方社区 PR 和赛事正式提交仍等待正确性、稳定吞吐复现及完整精度验证。运行代码、recipe 和分发配置沿用 manifest 标识的源码祖先；本轮仅更新共享状态文档。

2026-11-16 整理树没有在目标机运行，`submitted_tree_runtime_verified=false`、`model_accuracy_verified=false`、`full105_run=false`。完整 105 题尚未获本轮授权；历史成绩不能继承到本树。

- [源码与结果身份](manifest.json)
- [冻结实验历史吞吐摘要](historical-results.json)
- [配套 plugin 复现说明](https://github.com/braveloop/vllm-plugin-FL/blob/bravegpuwinner/s2-community-candidate/docs/competition/bravegpuwinner-s2.md)

保留 `bravegpuwinner/s2-bf16-reference` 作为成对 BF16 对照；当前分支为 `bravegpuwinner/s2-community-candidate`。没有创建官方社区 PR。
