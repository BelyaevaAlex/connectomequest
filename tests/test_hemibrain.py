import json

import pytest

from connectomequest.adapters.hemibrain import _parse_roi_info

ROI_INFO = {"AL(R)": {"pre": 3, "post": 5}}


@pytest.mark.parametrize(
    "value",
    [
        ROI_INFO,
        json.dumps(ROI_INFO),
        json.dumps(json.dumps(ROI_INFO)),
        None,
        "",
    ],
)
def test_parse_roi_info_accepts_neuprint_response_encodings(value: object) -> None:
    expected = {} if value in (None, "") else ROI_INFO
    assert _parse_roi_info(value) == expected


def test_parse_roi_info_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="must decode to an object"):
        _parse_roi_info("[1, 2, 3]")
