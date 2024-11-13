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

package graph

import (
	"github.com/pkg/errors"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gx-org/backend/graph"
	"github.com/gx-org/backend/platform"
	"github.com/gx-org/backend/shape"
	pjrtplatform "github.com/gx-org/xlapjrt/backend/platform"
	pjrtgx "github.com/gx-org/xlapjrt"
)

type nodeRunner struct {
	device *pjrtplatform.Device
	graph  *Graph
}

func bufferShape(buffer *pjrt.Buffer) (*shape.Shape, error) {
	dtype, err := buffer.DType()
	if err != nil {
		return nil, err
	}
	dims, err := buffer.Dimensions()
	if err != nil {
		return nil, err
	}
	return &shape.Shape{
		DType:       pjrtgx.ToGXDType(dtype),
		AxisLengths: dims,
	}, nil
}

func checkShape(got, want *shape.Shape) error {
	if got.DType != want.DType {
		return errors.Errorf("PJRT backend returned a buffer with a %s data type but GX expects a %s data type", got.DType, want.DType)
	}
	if got.Size() != want.Size() {
		return errors.Errorf("PJRT backend returned a buffer with axis lengths %v but GX expects %v", got.AxisLengths, want.AxisLengths)
	}
	return nil
}

func toHandles(dev *pjrtplatform.Device, buffers []*pjrt.Buffer, shapes []*shape.Shape) ([]platform.DeviceHandle, error) {
	handles := make([]platform.DeviceHandle, len(buffers))
	for i, buffer := range buffers {
		bufferShape, err := bufferShape(buffer)
		if err != nil {
			return nil, err
		}
		expectedShape := shapes[i]
		if err := checkShape(bufferShape, expectedShape); err != nil {
			return nil, err
		}
		handles[i], err = pjrtplatform.NewHandle(dev, buffer, expectedShape)
		if err != nil {
			return nil, err
		}
	}
	return handles, nil
}

// newNodeRunner returns a new node runner given a function and a graph.
func (graph *Graph) newNodeRunner(dev *pjrtplatform.Device) graph.Runner {
	return &nodeRunner{device: dev, graph: graph}
}
func (r *nodeRunner) Run(args []platform.Handle) (out, traced []platform.DeviceHandle, err error) {
	deviceBuffers := make([]*pjrt.Buffer, len(args))
	for i, arg := range args {
		deviceBuffers[i] = arg.(*pjrtplatform.Handle).OnDeviceBuffer()
	}
	results, err := r.graph.Executable().Execute(deviceBuffers...).Done()
	if err != nil {
		return nil, nil, err
	}
	outShapes := r.graph.OutShapes()
	numOut := len(outShapes)
	out, err = toHandles(r.device, results[:numOut], outShapes)
	if err != nil {
		return nil, nil, err
	}
	traced, err = toHandles(r.device, results[numOut:], r.graph.TracedShapes())
	if err != nil {
		return nil, nil, err
	}
	return out, traced, nil
}
