from errors import MissingDataError, to_issue


def test_known_application_error_preserves_user_message_and_details() -> None:
    issue = to_issue(MissingDataError("Missing prices.", "provider response was empty"))

    assert issue.category == "missing_data"
    assert issue.message == "Missing prices."
    assert issue.details == "provider response was empty"


def test_unexpected_error_is_wrapped() -> None:
    issue = to_issue(RuntimeError("boom"))

    assert issue.category == "unexpected_error"
    assert issue.message == "Unexpected error during analysis."
    assert issue.details == "boom"
