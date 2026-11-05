# 续跑记录
此前会话明确报 usage limit。恢复时仅根 agent 可见；b01/b02/b06/b07/b11/b12 六批已完成。b03/b08/b13 仅留 start_1.txt，无原始回复、无返回的生成器 ID，也无法确认是否已成功派发。为避免隐瞒失败或超出上限，保守记为一次 infra_error，再派发记 attempt_2。agent_id 标为 unavailable_after_usage_limit_interruption，不能视为已恢复实际 ID；end 为恢复检查时间，不是准确失败时刻。该追踪缺失需 Claude 在协议审核中显式处理，不能仅凭字段非空宣称完整可追溯。
本轮不重跑已合格六批，不改提示词，不扩大重试预算。其他未启动批次从 attempt_1 开始。
