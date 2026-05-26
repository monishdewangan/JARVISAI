from datetime import datetime, timezone, timedelta
import time as _time

def get_time_information() -> str:
    # Use local time with UTC offset so the AI is aware of the actual timezone
    utc_offset_seconds = -_time.timezone if _time.daylight == 0 else -_time.altzone
    local_tz = timezone(timedelta(seconds=utc_offset_seconds))
    now = datetime.now(tz=local_tz)
    return now.strftime("%A, %B %d, %Y, %I:%M %p %Z")
