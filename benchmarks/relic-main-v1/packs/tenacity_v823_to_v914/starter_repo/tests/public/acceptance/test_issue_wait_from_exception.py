"""Acceptance check for issue_wait_from_exception.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: `wait_exception(predicate)` calls the predicate once per failed attempt, hands it the raised exception, and uses the number it returns as that attempt's pause; chained and fixed waits behave exactly as before.
"""
from tenacity import retry, stop_after_attempt, wait_exception


def test_wait_exception_uses_each_raised_exception_for_its_delay():
    errors = [RuntimeError("429"), RuntimeError("503")]
    delays = {"429": 3, "503": 12}
    received = []
    pauses = []
    attempts = [0]

    def predicate(error):
        received.append(error)
        return delays[error.args[0]]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exception(predicate),
        sleep=pauses.append,
    )
    def request():
        if attempts[0] < len(errors):
            error = errors[attempts[0]]
            attempts[0] += 1
            raise error
        return "ok"

    assert request() == "ok"
    assert received[0] is errors[0]
    assert received[1] is errors[1]
    assert len(received) == 2
    assert [float(pause) for pause in pauses] == [3, 12]
