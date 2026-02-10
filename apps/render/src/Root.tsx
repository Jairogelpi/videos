import "./index.css";
import { Composition } from "remotion";
import { MyComposition } from "./Composition";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="LyricVideo"
        component={MyComposition}
        durationInFrames={1800} // 60s @ 30fps
        fps={30}
        width={1080}
        height={1920}
      />
    </>
  );
};
