from string import Template

MODEL_TEMPLATE = Template(
    """$background

# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text
"""
)

JUDGER_TEMPLATE = Template(
    """# NPC Profile:
## Name
$name_text

## Title
$title

## Description
$description

## Definition
$definition_text

## Long Definition
$long_definition_text

You are an AI NPC system. You need to simulate a user and interact with AI NPC. For each round, You should give your response to AI NPC. It will be in a JSON format: {"winner": "model_a", "next_round_user_speaks": "YOUR RESPONSE AS THE SIMULATED USER", "decision_reason": "None"}.
"""
)

MODEL_TEMPLATE_ZH = Template(
    """$background

# NPC 角色档案:
## 姓名
$name_text

## 称号
$title

## 描述
$description

## 定义
$definition_text

## 长定义
$long_definition_text
"""
)

JUDGER_TEMPLATE_ZH = Template(
    """# NPC 角色档案:
## 姓名
$name_text

## 称号
$title

## 描述
$description

## 定义
$definition_text

## 长定义
$long_definition_text

你是一个 AI NPC 系统。你需要模拟一个用户并与 AI NPC 进行互动。在每一轮中，你需要对 AI NPC 给出你的响应。响应格式将是一个 JSON 格式，例如：
{"winner": "model_a", "next_round_user_speaks": "作为模拟用户的你的回应内容", "decision_reason": "无"}。
"""
)


TEMPLATES = {
    "en": {
        "model": MODEL_TEMPLATE,
        "judger": JUDGER_TEMPLATE,
    },
    "zh": {
        "model": MODEL_TEMPLATE_ZH,
        "judger": JUDGER_TEMPLATE_ZH,
    },
}

def get_template(language="zh"):
    """
    根据模板类型和语言提取模板。

    参数:
        template_type (str): 模板类型，例如 'model' 或 'judger'。
        language (str): 语言，例如 'en' 或 'zh'。
    
    返回:
        Template: 返回对应的模板对象。
    """
    import json
    templates_by_language = TEMPLATES.get(language, {})
    return templates_by_language