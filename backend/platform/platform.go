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

// Package platform provides the pjrt platform for GX.
package platform

import (
	"fmt"

	"github.com/pkg/errors"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gopjrt/pjrt"
	pjrtgx "github.com/gx-org/xlapjrt"
)

// Platform is the PJRT backend.
type Platform struct {
	bck       compute.Backend
	clt       *pjrt.Client
	finalized bool
}

// New PJRT backend.
func New(clt *pjrt.Client, bck compute.Backend) *Platform {
	return &Platform{clt: clt, bck: bck}
}

// Backend returns the backend owning the platform.
func (plat *Platform) Backend() compute.Backend {
	return plat.bck
}

// Name of the backend.
func (plat *Platform) Name() string {
	return "pjrt"
}

// String returns the same as Name.
func (plat *Platform) String() string {
	return plat.Name()
}

// Description is a longer description of the Backend that can be used to pretty-print.
func (plat *Platform) Description() string {
	if !plat.clt.IsValid() {
		return "invalid PJRT client"
	}
	return plat.clt.String()
}

// NumDevices returns the number of devices available for this Backend.
func (plat *Platform) NumDevices() int {
	if !plat.clt.IsValid() {
		return 0
	}
	return len(plat.clt.AddressableDevices())
}

// DeviceDescription returns a description of the device at the given deviceNum.
func (plat *Platform) DeviceDescription(deviceNum compute.DeviceNum) string {
	if !plat.clt.IsValid() {
		return "invalid PJRT client"
	}
	devices := plat.clt.AddressableDevices()
	if int(deviceNum) < 0 || int(deviceNum) >= len(devices) {
		return fmt.Sprintf("invalid deviceNum %d", deviceNum)
	}
	desc, err := devices[deviceNum].GetDescription()
	if err != nil {
		return fmt.Sprintf("failed to get description for device %d: %v", deviceNum, err)
	}
	return fmt.Sprintf("%s [processId=%d]", desc.DebugString(), desc.ProcessIndex())
}

// Capabilities returns information about what is supported by this backend.
func (plat *Platform) Capabilities() compute.Capabilities {
	return compute.Capabilities{}
}

// BufferFromFlatData transfers data from Go given as a flat slice to the deviceNum, and returns the corresponding Buffer.
func (plat *Platform) BufferFromFlatData(deviceNum compute.DeviceNum, flat any, shape shapes.Shape) (compute.Buffer, error) {
	data := dtypes.UnsafeByteSliceFromAny(flat)
	return plat.send(deviceNum, data, shape)
}

// HasSharedBuffers returns whether this PJRT plugin supports shared buffers.
func (plat *Platform) HasSharedBuffers() bool {
	return false
}

// NewSharedBuffer returns a shared buffer that can be both used as input for execution of computations and directly read or mutated by the clients.
func (plat *Platform) NewSharedBuffer(deviceNum compute.DeviceNum, shape shapes.Shape) (buffer compute.Buffer, flat any, err error) {
	devices := plat.clt.AddressableDevices()
	if int(deviceNum) < 0 || int(deviceNum) >= len(devices) {
		return nil, nil, errors.Errorf("deviceNum=%d not available for backend, only %d devices are available", deviceNum, len(devices))
	}
	dt := pjrtgx.ToPJDType(shape.DType)
	pjrtBuffer, _, err := plat.clt.NewSharedBuffer(dt, shape.Dimensions, devices[deviceNum])
	if err != nil {
		return nil, nil, err
	}
	h, err := NewHandle(plat, deviceNum, pjrtBuffer, shape)
	if err != nil {
		return nil, nil, err
	}
	flat, err = h.Data()
	if err != nil {
		return nil, nil, err
	}
	return h, flat, nil
}

// Client returns the PJRT client.
func (plat *Platform) Client() *pjrt.Client {
	return plat.clt
}

// Finalize releases all the associated resources immediately, and makes the backend invalid.
func (plat *Platform) Finalize() {
	if plat.finalized {
		return
	}
	plat.finalized = true
	if plat.clt != nil {
		_ = plat.clt.Destroy()
	}
}

// IsFinalized returns true if the backend is in an invalid state.
func (plat *Platform) IsFinalized() bool {
	return plat == nil || plat.finalized || !plat.clt.IsValid()
}

func toInt32(input []int) []int32 {
	result := make([]int32, len(input))
	for i, n := range input {
		result[i] = int32(n)
	}
	return result
}

func toInt(input []int32) []int {
	result := make([]int, len(input))
	for i, n := range input {
		result[i] = int(n)
	}
	return result
}
