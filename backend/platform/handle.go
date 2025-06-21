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
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/backend/shape"
)

type (
	// Handle of a PJRT buffer.
	Handle struct {
		device *Device
		buffer *pjrt.Buffer
		shape  *shape.Shape
	}

	// PJRTLiteral extracts literal values from handles to create XLA constants.
	PJRTLiteral interface {
		Literal() *xlabuilder.Literal
	}
)

var _ platform.DeviceHandle = (*Handle)(nil)

// NewHandle returns a new platform handle given a PJRT buffer.
func NewHandle(dev *Device, buffer *pjrt.Buffer, sh *shape.Shape) (*Handle, error) {
	return &Handle{
		device: dev,
		buffer: buffer,
		shape:  sh,
	}, nil
}

// Shape of the underlying array.
func (h *Handle) Shape() *shape.Shape {
	return h.shape
}

// OnDeviceBuffer returns the PJRT buffer.
func (h *Handle) OnDeviceBuffer() *pjrt.Buffer {
	return h.buffer
}

// ToDevice transfers the handle to a device.
func (h *Handle) ToDevice(dev platform.Device) (platform.DeviceHandle, error) {
	pjrtDev, ok := dev.(*Device)
	if ok {
		return ToDevice(pjrtDev, h)
	}
	return nil, errors.Errorf("not implemented")
}

func (h *Handle) toDevice(dev *Device) (*Handle, error) {
	if h.device == dev {
		return h, nil
	}
	data := make([]byte, h.shape.ByteSize())
	if err := h.buffer.ToHost(data); err != nil {
		return nil, err
	}
	return dev.send(data, h.Shape())
}

// ToHost fetches the data from the handle and write it to buffer.
func (h *Handle) ToHost(buf platform.HostBuffer) error {
	data := buf.Acquire()
	defer buf.Release()
	return h.buffer.ToHost(data)
}

// Device on which the array is located.
func (h *Handle) Device() platform.Device {
	return h.device
}

// String representation of the handle.
func (h *Handle) String() string {
	return fmt.Sprintf("PJRT %T: %s", h, h.shape.String())
}

// ToDevice sends a generic handle to a device.
func ToDevice(dev *Device, handle platform.Handle) (*Handle, error) {
	switch handleT := handle.(type) {
	case *Handle:
		return handleT.toDevice(dev)
	case platform.HostBuffer:
		return dev.sendFromHost(handleT)
	}
	return nil, errors.Errorf("not implemented")
}
