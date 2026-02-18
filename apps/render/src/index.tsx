import React from 'react';
import { registerRoot, Composition } from 'remotion';
import { MyComposition } from './Composition';
import './index.css';

export const RemotionRoot: React.FC = () => {
    return (
        <>
        <Composition
        id= "LyricVideo"
    component = { MyComposition }
    durationInFrames = { 30 * 60 } // Default duration, overridden by inputProps
    fps = { 30}
    width = { 1080}
    height = { 1920} // 9:16 Vertical Video
    defaultProps = {{
        words: [],
            audioUrl: '',
                videoBgUrl: '',
                    motion_manifest: undefined
    }
}
      />
    </>
  );
};

registerRoot(RemotionRoot);
