using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Apt.Unity.Projection
{
    public class AutoMoveTracker : TrackerBase
    {
        [Min(0.001f)]
        public float BoundsSize = 2;

        [Range(0, 1)]
        public float XMovement = 0.5f;
        [Range(0, 1)]
        public float YMovement = 0.3f;
        [Range(0, 1)]
        public float ZMovement = 0;

        private float HalfBoundSize => BoundsSize * 0.5f;

        void Start()
        {
            IsTracking = true;
        }

        void Update()
        {
            if(Input.GetKeyUp(KeyCode.A) && (Input.GetKey(KeyCode.LeftControl) || Input.GetKey(KeyCode.RightControl)))
            {
                IsTracking = !IsTracking;
                SecondsHasBeenTracked = 0;
            }

            if(IsTracking)
            {
                SecondsHasBeenTracked += Time.deltaTime;
                float xSize = XMovement * HalfBoundSize;
                float ySize = YMovement * HalfBoundSize;
                float zSize = ZMovement * HalfBoundSize;
                translation.x =  Mathf.Sin(SecondsHasBeenTracked) * xSize;
                translation.y =  Mathf.Sin(SecondsHasBeenTracked - (Mathf.PI * 2 / 3)) * ySize;
                translation.z =  Mathf.Sin(SecondsHasBeenTracked) * zSize;
            }

        }
    }
}
