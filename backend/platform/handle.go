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
	"unsafe"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
)

type (
	// Handle of a PJRT buffer.
	Handle struct {
		plat   *Platform
		device compute.DeviceNum
		buffer *pjrt.Buffer
		shape  shapes.Shape
	}

	// PJRTLiteral extracts literal values from handles to create XLA constants.
	PJRTLiteral interface {
		Literal() *xlabuilder.Literal
	}
)

var _ compute.Buffer = (*Handle)(nil)

// NewHandle returns a new platform handle given a PJRT buffer.
func NewHandle(plat *Platform, dev compute.DeviceNum, buffer *pjrt.Buffer, sh shapes.Shape) (*Handle, error) {
	return &Handle{
		plat:   plat,
		device: dev,
		buffer: buffer,
		shape:  sh,
	}, nil
}

// Backend returns the backend that owns this buffer.
func (h *Handle) Backend() compute.Backend {
	return h.plat.Backend()
}

// Finalize allows the client to inform the backend that the buffer is no longer needed.
func (h *Handle) Finalize() error {
	return h.buffer.Destroy()
}

// Shape returns the shape for the buffer.
func (h *Handle) Shape() (shapes.Shape, error) {
	return h.shape, nil
}

// OnDeviceBuffer returns the PJRT buffer.
func (h *Handle) OnDeviceBuffer() *pjrt.Buffer {
	return h.buffer
}

// CopyToDevice copies the buffer to another device on the same backend.
func (h *Handle) CopyToDevice(dev compute.DeviceNum) (compute.Buffer, error) {
	return h.toDevice(dev)
}

func (h *Handle) toDevice(dev compute.DeviceNum) (*Handle, error) {
	if h.device == dev {
		return h, nil
	}
	data := make([]byte, int(h.shape.ByteSize()))
	if err := h.buffer.ToHost(data); err != nil {
		return nil, err
	}
	return h.plat.send(dev, data, h.shape)
}

// ToFlatData transfers the flat values of the buffer to the Go flat array.
func (h *Handle) ToFlatData(flat any) error {
	if h.shape.IsZeroSize() {
		return nil
	}
	buf := dtypes.UnsafeByteSliceFromAny(flat)
	return h.buffer.ToHost(buf)
}

// Data returns a slice pointing to the buffer storage memory directly.
func (h *Handle) Data() (flat any, err error) {
	rawStorage, err := h.buffer.UnsafePointer()
	if err != nil {
		return nil, err
	}
	return dtypes.UnsafeAnySliceFromBytes(rawStorage, h.shape.DType, h.shape.Size()), nil
}

// DeviceNum returns the deviceNum for the buffer.
func (h *Handle) DeviceNum() (compute.DeviceNum, error) {
	return h.device, nil
}

// String representation of the handle.
func (h *Handle) String() string {
	return fmt.Sprintf("PJRT %T: %s", h, h.shape.String())
}

// ToDevice sends a generic buffer to a device.
func ToDevice(plat *Platform, dev compute.DeviceNum, handle compute.Buffer) (*Handle, error) {
	if handleT, ok := handle.(*Handle); ok && handleT.plat == plat {
		return handleT.toDevice(dev)
	}
	sh, err := handle.Shape()
	if err != nil {
		return nil, err
	}
	raw := make([]byte, sh.ByteSize())
	flat := dtypes.UnsafeAnySliceFromBytes(unsafe.Pointer(unsafe.SliceData(raw)), sh.DType, sh.Size())
	if err := handle.ToFlatData(flat); err != nil {
		return nil, err
	}
	return plat.send(dev, raw, sh)
}
