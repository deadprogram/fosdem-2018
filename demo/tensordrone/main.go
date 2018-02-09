/*
How to setup
You must be using a PS3 or compatible controller, along with
any of the Parrot Minidrone drones to run this example.

You run the Go program on your computer and communicate
wirelessly with the Parrot Minidrone.

You must also have the camera on the drone.

How to run

	go run tensordrone/main.go "Mambo_1234" dualshock3.json 0 tensorflow_inception_graph.pb imagenet_comp_graph_label_strings.txt

NOTE: sudo is required to use BLE in Linux
*/

package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"os"
	"strconv"
	"sync/atomic"
	"time"

	"gobot.io/x/gobot"
	"gobot.io/x/gobot/platforms/ble"
	"gobot.io/x/gobot/platforms/joystick"
	"gobot.io/x/gobot/platforms/opencv"
	"gobot.io/x/gobot/platforms/parrot/minidrone"
	"gocv.io/x/gocv"
)

type pair struct {
	x float64
	y float64
}

var leftX, leftY, rightX, rightY atomic.Value

const offset = 32767.0

func main() {
	// parse args
	if len(os.Args) < 6 {
		fmt.Println("How to run:\n\ttensordrone [drone ID] [joystick JSON file] [cameraid] [modelfile] [descriptionsfile]")
		return
	}

	droneID := os.Args[1]
	joystickFile := os.Args[2]
	deviceID, _ := strconv.Atoi(os.Args[3])
	model := os.Args[4]
	descriptions, _ := readDescriptions(os.Args[5])

	joystickAdaptor := joystick.NewAdaptor()
	stick := joystick.NewDriver(joystickAdaptor, joystickFile)

	droneAdaptor := ble.NewClientAdaptor(droneID)
	drone := minidrone.NewDriver(droneAdaptor)

	window := opencv.NewWindowDriver()
	camera := opencv.NewCameraDriver(deviceID)

	// open Tensorflow DNN classifier
	net := gocv.ReadNetFromTensorflow(model)
	defer net.Close()

	work := func() {
		leftX.Store(float64(0.0))
		leftY.Store(float64(0.0))
		rightX.Store(float64(0.0))
		rightY.Store(float64(0.0))

		camera.On(opencv.Frame, func(data interface{}) {
			img := data.(gocv.Mat)

			// convert image Mat to 224x244 blob that the classifier can analyze
			blob := gocv.BlobFromImage(img, 1.0, image.Pt(224, 244), gocv.NewScalar(0, 0, 0, 0), true, false)

			// feed the blob into the Tensorflow classifier network
			net.SetInput(blob, "input")

			// run a forward pass thru the network
			prob := net.Forward("softmax2")

			// reshape the results into a 1x1000 matrix
			probMat := prob.Reshape(1, 1)

			// determine the most probable classification, which will be max value
			_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)

			// display classification based on position in the descriptions file
			desc := "Unknown"
			if maxLoc.X < 1000 {
				desc = descriptions[maxLoc.X]
			}
			status := fmt.Sprintf("description: %v, maxVal: %v\n", desc, maxVal)
			gocv.PutText(img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)

			blob.Close()
			prob.Close()
			probMat.Close()

			window.ShowImage(img)
			window.WaitKey(1)
		})

		stick.On(joystick.SquarePress, func(data interface{}) {
			drone.Stop()
		})

		stick.On(joystick.TrianglePress, func(data interface{}) {
			drone.HullProtection(true)
			drone.TakeOff()
		})

		stick.On(joystick.XPress, func(data interface{}) {
			drone.Land()
		})

		stick.On(joystick.LeftX, func(data interface{}) {
			val := float64(data.(int16))
			leftX.Store(val)
		})

		stick.On(joystick.LeftY, func(data interface{}) {
			val := float64(data.(int16))
			leftY.Store(val)
		})

		stick.On(joystick.RightX, func(data interface{}) {
			val := float64(data.(int16))
			rightX.Store(val)
		})

		stick.On(joystick.RightY, func(data interface{}) {
			val := float64(data.(int16))
			rightY.Store(val)
		})

		gobot.Every(10*time.Millisecond, func() {
			rightStick := getRightStick()

			switch {
			case rightStick.y < -10:
				drone.Forward(minidrone.ValidatePitch(rightStick.y, offset))
			case rightStick.y > 10:
				drone.Backward(minidrone.ValidatePitch(rightStick.y, offset))
			default:
				drone.Forward(0)
			}

			switch {
			case rightStick.x > 10:
				drone.Right(minidrone.ValidatePitch(rightStick.x, offset))
			case rightStick.x < -10:
				drone.Left(minidrone.ValidatePitch(rightStick.x, offset))
			default:
				drone.Right(0)
			}
		})

		gobot.Every(10*time.Millisecond, func() {
			leftStick := getLeftStick()
			switch {
			case leftStick.y < -10:
				drone.Up(minidrone.ValidatePitch(leftStick.y, offset))
			case leftStick.y > 10:
				drone.Down(minidrone.ValidatePitch(leftStick.y, offset))
			default:
				drone.Up(0)
			}

			switch {
			case leftStick.x > 20:
				drone.Clockwise(minidrone.ValidatePitch(leftStick.x, offset))
			case leftStick.x < -20:
				drone.CounterClockwise(minidrone.ValidatePitch(leftStick.x, offset))
			default:
				drone.Clockwise(0)
			}
		})
	}

	robot := gobot.NewRobot("tensordrone",
		[]gobot.Connection{joystickAdaptor, droneAdaptor},
		[]gobot.Device{stick, drone, window, camera},
		work,
	)

	robot.Start()
}

func getLeftStick() pair {
	s := pair{x: 0, y: 0}
	s.x = leftX.Load().(float64)
	s.y = leftY.Load().(float64)
	return s
}

func getRightStick() pair {
	s := pair{x: 0, y: 0}
	s.x = rightX.Load().(float64)
	s.y = rightY.Load().(float64)
	return s
}

// readDescriptions reads the descriptions from a file
// and returns a slice of its lines.
func readDescriptions(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return lines, scanner.Err()
}
