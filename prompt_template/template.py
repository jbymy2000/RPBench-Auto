import Template

TEMPLATE = Template(
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