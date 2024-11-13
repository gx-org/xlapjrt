// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package platform

import (
	"github.com/pkg/errors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/backend/shape"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Device is a PJRT device.
type Device struct {
	plat *Platform
	ord  int
}

// Platform owning the device.
func (dev *Device) Platform() platform.Platform {
	return dev.plat
}

// Ordinal of the device on the platform.
func (dev *Device) Ordinal() int {
	return dev.ord
}

// Send raw data to the device. Return a handle from this package.
func (dev *Device) send(data []byte, sh *shape.Shape) (*Handle, error) {
	dt := pjrtgx.ToDType(sh.DType)
	if dt == dtypes.InvalidDType {
		return nil, errors.Errorf("GX %s data type not supported by pjrt", sh.DType.String())
	}
	buffer, err := dev.plat.clt.BufferFromHost().FromRawData(data, dt, sh.AxisLengths).Done()
	if err != nil {
		return nil, err
	}
	return NewHandle(dev, buffer, sh)
}

func (dev *Device) sendFromHost(handle platform.HostBuffer) (*Handle, error) {
	data := handle.Acquire()
	defer handle.Release()
	return dev.send(data, handle.Shape())
}

// Send raw data to the device.
func (dev *Device) Send(data []byte, sh *shape.Shape) (platform.DeviceHandle, error) {
	return dev.send(data, sh)
}
