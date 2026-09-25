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
	"fmt"

	"github.com/pkg/errors"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend"
	"github.com/gx-org/gx/golang/backend/kernels"
)

type (
	// Handle of a PJRT buffer.
	Handle struct {
		plat   *Platform
		device backend.DeviceNum
		buffer *pjrt.Buffer
		shape  shapes.Shape
	}

	// PJRTLiteral extracts literal values from handles to create XLA constants.
	PJRTLiteral interface {
		Literal() *xlabuilder.Literal
	}
)

var _ backend.DeviceHandle = (*Handle)(nil)

// NewHandle returns a new platform handle given a PJRT buffer.
func NewHandle(plat *Platform, dev backend.DeviceNum, buffer *pjrt.Buffer, sh shapes.Shape) (*Handle, error) {
	return &Handle{
		plat:   plat,
		device: dev,
		buffer: buffer,
		shape:  sh,
	}, nil
}

// Shape of the underlying array.
func (h *Handle) Shape() shapes.Shape {
	return h.shape
}

// OnDeviceBuffer returns the PJRT buffer.
func (h *Handle) OnDeviceBuffer() *pjrt.Buffer {
	return h.buffer
}

// ToDevice transfers the handle to a device.
func (h *Handle) ToDevice(dev backend.DeviceNum) (backend.DeviceHandle, error) {
	return h.toDevice(dev)
}

func (h *Handle) toDevice(dev backend.DeviceNum) (*Handle, error) {
	if h.device == dev {
		return h, nil
	}
	data := make([]byte, int(h.shape.ByteSize()))
	if err := h.buffer.ToHost(data); err != nil {
		return nil, err
	}
	return h.plat.send(dev, data, h.Shape())
}

// ToHost fetches the data from the handle and write it to buffer.
func (h *Handle) ToHost(buf []byte) error {
	return h.buffer.ToHost(buf)
}

// Device on which the array is located.
func (h *Handle) Device() backend.DeviceNum {
	return h.device
}

// String representation of the handle.
func (h *Handle) String() string {
	return fmt.Sprintf("PJRT %T: %s", h, h.shape.String())
}

// ToDevice sends a generic handle to a device.
func ToDevice(plat *Platform, dev backend.DeviceNum, handle backend.Handle) (*Handle, error) {
	switch handleT := handle.(type) {
	case *Handle:
		return handleT.toDevice(dev)
	case kernels.HostBuffer:
		return plat.sendFromHost(dev, handleT)
	}
	return nil, errors.Errorf("not implemented")
}
