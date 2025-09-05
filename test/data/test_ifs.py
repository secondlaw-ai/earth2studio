# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import shutil
import xarray as xr
import numpy as np
import pytest
from datetime import datetime, timedelta
from earth2studio.data import IFS
from earth2studio.models.px.fcn3 import VARIABLES

def now6h():
    """Get closest 6 hour timestamp"""
    nt = datetime.now()
    delta_hr = nt.hour % 6
    return datetime(nt.year, nt.month, nt.day, nt.hour - delta_hr)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
        [
            now6h() - timedelta(days=1),
            now6h() - timedelta(days=1, hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp"]])
def test_ifs_fetch(time, variable):
    ds = IFS(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert IFS.available(time[0])
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([now6h() - timedelta(days=2)], dtype="datetime64"),
    ],
)
@pytest.mark.parametrize("variable", [["u10m", "tcwv"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ifs_cache(time, variable, cache):

    ds = IFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cahce should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass

@pytest.mark.skip(reason="Fails...")
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        now6h() - timedelta(days=5),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_available(time, variable):
    assert not IFS.available(time)
    with pytest.raises(ValueError):
        ds = IFS()
        ds(time, variable)


    

def test_ifs_fetch_with_mp():
    time = np.array([datetime(2025, 9, 1)])
    ds0 = IFS(max_workers=0, cache=False)
    res0 = ds0(time, ["u10m", "v10m"])
    ds1 = IFS(max_workers=4, cache=False)
    res1 = ds1(time, ["u10m", "v10m"])
    assert res0.shape == res1.shape
    assert not np.isnan(res0.values).any()
    assert not np.isnan(res1.values).any()
    xr.testing.assert_equal(res0, res1)

@pytest.mark.skip(reason="Test speed of IC fetching.")
def test_ifs_fetch_with_mp_and_pressure_levels():
    time = np.array([datetime(2025, 9, 1)])
    ds0 = IFS(max_workers=16, cache=False)
    ds0(time, VARIABLES)
