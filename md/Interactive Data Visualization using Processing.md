# Interactive Data Visualization using Processing

## Download and install Processing

1. Go to https://processing.org, download the zip for Windows 64;
2. Extract it, then run the processing.exe;

## Basic graphic functions

``` processing
size(600, 400);               // set width and height for canvas
background(#666666);          // you could also use RGB values like background(64, 86, 76)
smooth();

stroke(#607F9C);              // set stroke color and width
strokeWeight(20);
point(100, 100);              // draw a point at (x, y)

stroke(#791F33);
line(100, 50, 100, 150);      // start x, y, end x, y

noStroke();
fill(#8FA89B);
ellipse(100, 100, 150, 100)   // center x, y, width, height

// x, y, width, height, start, end
// 0 deg is at right hand x axis horizontal, clockwise
arc(100, 100, 75, 75, 0, PI*0.5) // start from 0 deg to 90 deg
fill(0, 191, 255);
noStroke();
arc(367, 100, 75, 75, 0, radians(270));

rectMode(CORNER);
rect(60, 60, 80, 80);         // left, top, width, height

rectMode(CENTER);
noStroke();
fill(#CC5C54);
rect(300, 100, 80, 80);       // center x, y, width, height

rectMode(CORNERS);
stroke(#F69162);
noFill();
rect(460, 60, 540, 140);      // left, top, right, bottom

stroke(#FDF6DD);
noFill();
quad(450, 50, 500, 100, 450, 150, 400, 100);

noStroke();
fill(#74AD92);
triangle(250, 50, 300, 150, 350, 50);

// polygon
stroke(#45718c);
beginShape();
vertex(400, 150);
vertex(350, 125);
vertex(350, 75);
vertex(400, 50);
vertex(450, 75);
vertex(450, 125);
endShape(CLOSE);

// curves
stroke(#475D1C);
curveTightness(-3);
curve(400, 300, 400, 100, 500, 100, 500, 300);

// Black curve
stroke(0);
strokeWeight(3);
curveTightness(0);
beginShape();
curveVertex(100, 100);
curveVertex(100, 100);
curveVertex(150, 150);
curveVertex(250,  50);
curveVertex(300,  10);
curveVertex(400, 190);
curveVertex(500, 100);
curveVertex(500, 100);
endShape();

stroke(#BA3D49);
strokeWeight(3);
bezier(350, 75, 500, 25, 500, 175, 350, 125);

```

## Variables

``` processing
// int variable
int x;  // Declared x
x = 10;  // Initialized x
//println(x);
println("x = " + x);

// float variable
float e = 2.71828;

// boolean variable
boolean switchVar = true;
switchVar = !switchVar;

// char variable
char charVar = 'V';

// byte variable
byte dozen = 12;

// color variable
color cherryBlossomPink = #FFB7C5;

// The are global variables
int x = 0;
int y = 50;
int z;

void setup() {
  size(600, 200);
  smooth();
  // This is a local variable
  color darkGray = #333333;
  background(darkGray);
  println("darkGray = #" + hex(darkGray, 6));
  float randomFloat1 = random(10);
  println("randomFloat1 = " + randomFloat1);
  z = int(random(11));
}

void draw() {
  color darkGray = #111111;
  background(darkGray);
}

// Array ---------------------
// Manually create an array
int[] a = {100, 200, 300, 400, 500};

// fill in an empty array
int[] b = new int[3];

int n = 1000;
float[] xTop = new float[n];
float[] xBottom = new float[n];

for(int i = 0; i < n; i++) {
  xTop[i] = random(50, 550);
  xBottom[i] = random(50, 550);
  line(xTop[i], 25, xBottom[i], 175);
}

// Original array
int[] a = {7, 0, 4};

// copy
int[] b = new int[a.length];
//println(b);
arrayCopy(a, b);

// sort
a = sort(a);

// reverse
int[] bRev = reverse(b);

// append
a = append(a, 8);

// splice
b = splice(b, 10, 2);

// concatenation
int[] c = concat(a, b);

// String ---------------------------
String deerHunterOntology = "This is this. It's not something else. This is this.";
String poemLines[] = loadStrings("Szymborska.txt");

String quote = "     To be or not to be.     ";
quote = trim(quote);

int a = 798;
String A = nf(a, 10); // 0000000798

```

### Load external palette

https://colorbrewer2.org/
https://paletton.com/
https://color.adobe.com/

### Use different fonts

https://processing.org/tutorials/typography

### Sharing

https://openprocessing.org
http://sketchpad.cc/
https://www.arduino.cc/
