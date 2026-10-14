# -*- coding: utf-8 -*-
"""
分块模块的回归测试。

背景：二次切分（split_long_section）会把含标题行的 content 交给
RecursiveCharacterTextSplitter，而 '\n' 是分隔符之一 —— 于是标题行
被单独切出一个只有 9 字符的"孤儿块"。

这个文件要锁住【修复后】的正确行为。
"""
from app.core.logger import logger
from app.import_process.agent.nodes.node_document_split import (
    split_long_section,
    DEFAULT_MAX_CONTENT_LENGTH, step_3_split_by_content, MIN_CONTENT_LENGTH,
)

logger.remove()   # 测试时静音，日志会淹没断言输出


# 一份"标题行 + 超长正文"的 section —— 正文没有内部换行，
# 所以 '\n' 只出现在标题行和正文之间
def make_long_section():
    return {
        "title": "## 三、充电说明",
        "content": "## 三、充电说明\n" + "充电时请使用原装适配器，并确保通风良好。" * 120,
        "file_title": "HAK180产品手册",
    }


# ============================================================================
# 测试 1：二次切分不应产生"标题孤儿块"
# ============================================================================
def test_no_orphan_title_block():
    """
    修复前：会切出 len=9 的块（只有标题行）
    修复后：每个子块都应含有正文内容
    """
    subs = split_long_section(make_long_section(),DEFAULT_MAX_CONTENT_LENGTH)
    x = [x for x in subs if len(x["content"])<50]
    assert not x,f"存在孤儿块{[(s['title'], len(s['content'])) for s in x]}"



# ============================================================================
# 测试 2：任何子块都不应超过 MAX_CONTENT_LENGTH
# ============================================================================
def test_no_block_exceeds_max_length():
    """
    修复前：合并后某个块会变成 2010 字符（超过 2000）
    修复后：所有子块都应 <= DEFAULT_MAX_CONTENT_LENGTH
    """
    subs = step_3_split_by_content([make_long_section()],DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
    length = [len(x["content"]) for x in subs]
    x = [x for x in length if x>DEFAULT_MAX_CONTENT_LENGTH]
    assert not x,f"超过块上限{x}"



# ============================================================================
# 测试 3（加分）：子块编号应该连续
# ============================================================================
def test_sub_block_numbers_are_contiguous():
    """
    修复前：编号是 [1, 3]（_2 被合并掉了）
    修复后：应是 [1, 2, ...]
    """
    subs = step_3_split_by_content([make_long_section()],DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
    num_list=[x["part"] for x in subs]
    for i,x in enumerate(num_list,start=1):
        assert x==i,f"编号不连接{num_list}"