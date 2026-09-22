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
	"github.com/gx-org/backend"
	"github.com/gx-org/backend/shape"
	"github.com/gx-org/gx/golang/backend/kernels"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// send raw data to the device. Return a handle from this package.
func (plat *Platform) send(dev backend.DeviceNum, data []byte, sh *shape.Shape) (*Handle, error) {
	dt := pjrtgx.ToDType(sh.DType)
	if dt == dtypes.InvalidDType {
		return nil, errors.Errorf("GX %s data type not supported by pjrt", sh.DType.String())
	}
	buffer, err := plat.clt.BufferFromHost().FromRawData(data, dt, sh.AxisLengths).Done()
	if err != nil {
		return nil, err
	}
	return NewHandle(plat, dev, buffer, sh)
}

func (plat *Platform) sendFromHost(dev backend.DeviceNum, handle kernels.HostBuffer) (*Handle, error) {
	data := handle.Acquire()
	defer handle.Release()
	return plat.send(dev, data, handle.Shape())
}
