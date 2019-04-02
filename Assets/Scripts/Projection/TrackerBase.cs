using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Apt.Unity.Projection
{
    public class TrackerBase : MonoBehaviour
    {
        [HideInInspector]
        public bool IsTracking { get; protected set; }
        [HideInInspector]
        public ulong TrackedId { get; protected set; }
        [HideInInspector]
        public Vector3 Translation { get => translation; }
        [HideInInspector]
        public float SecondsHasBeenTracked { get; protected set; }

        protected Vector3 translation;
    }
}
