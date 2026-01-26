"""iMessage sending via macOS AppleScript."""

import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def send_imessage(phone_number: str, message: str) -> bool:
    """Send an iMessage using macOS Messages app via AppleScript.

    Args:
        phone_number: Phone number to send to (e.g., "+15551234567")
        message: Message text to send

    Returns:
        True if message was sent successfully, False otherwise
    """
    # Escape quotes in message for AppleScript
    escaped_message = message.replace('"', '\\"').replace("'", "'\"'\"'")

    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{phone_number}" of targetService
        send "{escaped_message}" to targetBuddy
    end tell
    '''

    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            logger.error(f"AppleScript error: {result.stderr}")
            return False

        logger.info(f"iMessage sent to {phone_number}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("iMessage send timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to send iMessage: {e}")
        return False


def send_to_multiple(phone_numbers: list[str], message: str) -> dict:
    """Send iMessage to multiple recipients.

    Args:
        phone_numbers: List of phone numbers
        message: Message text to send

    Returns:
        Dict with 'success' and 'failed' lists
    """
    results = {"success": [], "failed": []}

    for number in phone_numbers:
        if send_imessage(number, message):
            results["success"].append(number)
        else:
            results["failed"].append(number)

    return results


def format_boss_message(
    boss_name: str,
    negations: dict,
    status_resistances: dict,
    is_nightlord: bool = True
) -> str:
    """Format boss weaknesses/strengths into a clean text message.

    Args:
        boss_name: Display name of the boss
        negations: Dict of damage type -> value
        status_resistances: Dict of status type -> value/array/"immune"
        is_nightlord: True for nightlord, False for field boss

    Returns:
        Formatted message string
    """
    # Header
    emoji = "🌙" if is_nightlord else "⚔️"
    boss_type = "NIGHTLORD" if is_nightlord else "FIELD BOSS"
    lines = [f"{emoji} {boss_type}: {boss_name}", ""]

    # Damage negations - separate into weaknesses and strengths
    weaknesses = []
    strengths = []

    # Abbreviations for damage types
    abbrev = {
        "standard": "Std",
        "slash": "Sla",
        "strike": "Str",
        "pierce": "Prc",
        "magic": "Mag",
        "fire": "Fir",
        "lightning": "Lit",
        "holy": "Hol"
    }

    for damage_type, value in negations.items():
        if value == 0:
            continue
        short_name = abbrev.get(damage_type, damage_type[:3].capitalize())
        if value < 0:
            weaknesses.append(f"{short_name}: {value}%")
        else:
            strengths.append(f"{short_name}: +{value}%")

    # Format damage negations in two columns (weakness | strength)
    if weaknesses or strengths:
        # Pair up weaknesses and strengths side by side
        max_rows = max(len(weaknesses), len(strengths))
        for i in range(max_rows):
            weak_part = weaknesses[i] if i < len(weaknesses) else ""
            strong_part = strengths[i] if i < len(strengths) else ""

            if weak_part and strong_part:
                # Pad weakness to fixed width for alignment
                lines.append(f"{weak_part:<12} {strong_part}")
            elif weak_part:
                lines.append(weak_part)
            elif strong_part:
                lines.append(f"{'':<12} {strong_part}")

        lines.append("")

    # Status resistances
    # Default values for reference: [154, 252, 542, 999]
    status_abbrev = {
        "poison": "Poison",
        "scarlet_rot": "Rot",
        "blood_loss": "Blood",
        "frostbite": "Frost",
        "sleep": "Sleep",
        "madness": "Madness"
    }

    immune = []
    strong = []
    weak = []

    for status, value in status_resistances.items():
        name = status_abbrev.get(status, status)

        if value == "immune":
            immune.append(name)
        elif isinstance(value, list):
            # Check if it differs from default [154, 252, 542, 999]
            if len(value) == 4 and value[0] == 154:
                continue  # Default - don't show
            elif value[0] < 154:
                weak.append(name)
            elif len(value) < 4:
                strong.append(name)

    if immune:
        lines.append(f"Immune: {', '.join(immune)}")
    if strong:
        lines.append(f"Strong: {', '.join(strong)}")
    if weak:
        lines.append(f"Weak: {', '.join(weak)}")

    return "\n".join(lines)
