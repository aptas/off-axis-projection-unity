using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Apt.Unity.Projection
{
    public class KeyboardTracker : TrackerBase
    {
        [Min(0.001f)]
        public float BoundsSize = 2;
        [Tooltip("How much of the bounds size to move each second")]
        [Range(0, 1)]
        public float MoveAmountPerSecond = 0.2f;

        private Bounds bounds;

        void Start()
        {
            bounds = new Bounds(Vector3.zero, new Vector3(BoundsSize, BoundsSize, BoundsSize));
        }

        void Update()
        {
            if(Input.GetKeyUp(KeyCode.Space))
            {
                IsTracking = !IsTracking;
                SecondsHasBeenTracked = 0;
            }

            if(IsTracking)
            {
                SecondsHasBeenTracked += Time.deltaTime;

                float MovedThisFrame = MoveAmountPerSecond * BoundsSize * Time.deltaTime;
                if (Input.GetKey(KeyCode.A))
                {
                    translation.x = Mathf.Max(bounds.min.x, translation.x - MovedThisFrame);
                }
                if (Input.GetKey(KeyCode.D))
                {
                    translation.x = Mathf.Min(bounds.max.x, translation.x + MovedThisFrame);
                }
                if (Input.GetKey(KeyCode.W))
                {
                    translation.y = Mathf.Min(bounds.max.y, translation.y + MovedThisFrame);
                }
                if (Input.GetKey(KeyCode.S))
                {
                    translation.y = Mathf.Max(bounds.min.y, translation.y - MovedThisFrame);
                }
                if (Input.GetKey(KeyCode.Q))
                {
                    translation.z = Mathf.Max(bounds.min.z, translation.z - MovedThisFrame);
                }
                if (Input.GetKey(KeyCode.E))
                {
                    translation.z = Mathf.Min(bounds.max.z, translation.z + MovedThisFrame);
                }

                if (Input.GetKey(KeyCode.R))
                {
                    translation.Set(0, 0, 0);
                }

                if(Input.GetKey(KeyCode.N))
                {
                    TrackedId++;
                }
            }

        }
    }
}
