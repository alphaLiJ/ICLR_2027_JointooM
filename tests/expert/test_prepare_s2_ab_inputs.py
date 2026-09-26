from experiments.runners.prepare_s2_ab_inputs import (
    generate_official_style_warehouse,
)


def test_official_warehouse_geometry_matches_upstream_parameters():
    lines = generate_official_style_warehouse(
        num_wall_rows=5,
        num_wall_cols=2,
    ).splitlines()

    assert len(lines) == 23
    assert {len(line) for line in lines} == {23}
    assert sum(value == "." for line in lines for value in line) == 208
