import { Canvas, useFrame } from "@react-three/fiber";
import {
  Environment,
  Float,
  Html,
  Line,
  MeshTransmissionMaterial,
  OrbitControls,
  Sphere,
  Text,
} from "@react-three/drei";
import { Suspense, useMemo, useRef } from "react";
import * as THREE from "three";

function MarketRibbon({ reducedMotion, accentColor }) {
  const groupRef = useRef(null);
  const lineRefs = useRef([]);
  const waveSets = useMemo(() => {
    return new Array(4).fill(null).map((_, lineIndex) =>
      new Array(90).fill(null).map((__, pointIndex) => {
        const x = (pointIndex / 89) * 10 - 5;
        const phase = pointIndex * 0.45 + lineIndex * 0.8;
        const y = Math.sin(phase) * (0.22 + lineIndex * 0.045);
        const z = (lineIndex - 1.5) * 0.75 + Math.cos(phase * 0.65) * 0.18;
        return new THREE.Vector3(x, y, z);
      }),
    );
  }, []);

  useFrame((state) => {
    const elapsed = state.clock.getElapsedTime();

    if (groupRef.current) {
      groupRef.current.rotation.y = reducedMotion ? 0.18 : Math.sin(elapsed * 0.16) * 0.28 + 0.18;
      groupRef.current.rotation.x = reducedMotion ? -0.18 : Math.cos(elapsed * 0.13) * 0.08 - 0.18;
    }

    lineRefs.current.forEach((line, lineIndex) => {
      if (!line) {
        return;
      }
      const positions = line.geometry.attributes.position.array;
      for (let pointIndex = 0; pointIndex < 90; pointIndex += 1) {
        const baseIndex = pointIndex * 3;
        const x = (pointIndex / 89) * 10 - 5;
        const amplitude = 0.22 + lineIndex * 0.045;
        const phase = pointIndex * 0.45 + lineIndex * 0.8 + elapsed * (reducedMotion ? 0.18 : 0.72);
        positions[baseIndex] = x;
        positions[baseIndex + 1] = Math.sin(phase) * amplitude;
        positions[baseIndex + 2] =
          (lineIndex - 1.5) * 0.75 + Math.cos(phase * 0.6) * (reducedMotion ? 0.08 : 0.22);
      }
      line.geometry.attributes.position.needsUpdate = true;
    });
  });

  return (
    <group ref={groupRef}>
      {waveSets.map((points, index) => (
        <Line
          key={index}
          ref={(element) => {
            lineRefs.current[index] = element;
          }}
          points={points}
          color={index % 2 === 0 ? accentColor : "#8da17f"}
          transparent
          opacity={0.9 - index * 0.12}
          lineWidth={1.25}
        />
      ))}
      <mesh position={[0, -0.25, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[12, 8, 1, 1]} />
        <meshBasicMaterial color="#221915" transparent opacity={0.26} />
      </mesh>
    </group>
  );
}

function SignalOrbs({ reducedMotion }) {
  const groupRef = useRef(null);

  useFrame((state) => {
    if (!groupRef.current || reducedMotion) {
      return;
    }
    const elapsed = state.clock.getElapsedTime();
    groupRef.current.rotation.y = elapsed * 0.14;
    groupRef.current.children.forEach((child, index) => {
      child.position.y = Math.sin(elapsed * 0.8 + index * 0.7) * 0.35 + child.userData.baseY;
    });
  });

  return (
    <group ref={groupRef}>
      {[
        { position: [-2.2, 0.5, 1.2], color: "#3B82F6", scale: 0.24 },
        { position: [2.0, 1.2, -0.4], color: "#10B981", scale: 0.18 },
        { position: [1.2, -0.6, 1.7], color: "#60A5FA", scale: 0.12 },
      ].map((orb) => (
        <Float key={orb.color} speed={reducedMotion ? 0 : 1.4} rotationIntensity={0.2} floatIntensity={0.8}>
          <Sphere
            args={[orb.scale, 18, 18]}
            position={orb.position}
            userData={{ baseY: orb.position[1] }}
          >
            <meshStandardMaterial
              color={orb.color}
              emissive={orb.color}
              emissiveIntensity={0.45}
              roughness={0.22}
              metalness={0.45}
            />
          </Sphere>
        </Float>
      ))}
    </group>
  );
}

export default function HeroScene({ reducedMotion }) {
  return (
    <Canvas
      camera={{ position: [0, 1.1, 8.4], fov: 34 }}
      dpr={[1, 1.5]}
      className="hero-canvas"
    >
      <color attach="background" args={["#0B0F19"]} />
      <fog attach="fog" args={["#0B0F19", 9, 18]} />
      <ambientLight intensity={0.5} />
      <directionalLight position={[4, 6, 5]} intensity={2} color="#3B82F6" />
      <pointLight position={[-5, -1, 4]} intensity={1.5} color="#10B981" />
      <Suspense
        fallback={
          <Html center className="scene-loading">
            Calibrating field
          </Html>
        }
      >
        <group position={[0, 0.2, 0]}>
          <mesh position={[0, 0, 0]} rotation={[0.25, 0.15, -0.08]}>
            <torusKnotGeometry args={[1.2, 0.3, 160, 20]} />
            <MeshTransmissionMaterial
              color="#3B82F6"
              thickness={0.5}
              roughness={0.1}
              transmission={1}
              ior={1.2}
              chromaticAberration={0.06}
              backside
            />
          </mesh>
          <MarketRibbon reducedMotion={reducedMotion} accentColor="#3B82F6" />
          <SignalOrbs reducedMotion={reducedMotion} />
        </group>
        <Environment preset="city" />
      </Suspense>
      <OrbitControls
        enablePan={false}
        enableZoom={false}
        autoRotate={!reducedMotion}
        autoRotateSpeed={0.28}
        maxPolarAngle={Math.PI * 0.62}
        minPolarAngle={Math.PI * 0.34}
      />
    </Canvas>
  );
}
