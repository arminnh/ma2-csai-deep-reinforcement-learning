using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class CarTouchesTrack : MonoBehaviour {

	public bool touchesTrack;
	public bool resetCar;
	public int secondsOfFreedom = 5;
	private Vector3 startPos;
	private Quaternion startRotation;
	private float startTime = 0;

	// Use this for initialization
	void Start () {
		this.startPos = transform.position;
		this.startRotation = transform.rotation;

	}	
		
	void Update()
	{
		touchesTrack = IsGrounded();
		if (!touchesTrack && this.resetCar) {
			if (this.startTime == 0) {
				this.startTime = Time.time;
			}
		} 

		if (this.startTime != 0 && (Time.time - this.startTime >= this.secondsOfFreedom) ) {
			this.startTime = 0;
			transform.position = this.startPos;
			transform.rotation = this.startRotation;
			Rigidbody rb = GetComponent<Rigidbody> ();
			rb.velocity = new Vector3 (0, 0, 0);

		}
	}

	public bool IsGrounded()  
	{
		RaycastHit hit;
		if (Physics.Raycast (transform.position, -Vector3.up, out hit, 5)) {
			if (hit.collider != null && hit.collider.name == "GroundTrack")
				return true;
		}
		return false;
	}
		
}
